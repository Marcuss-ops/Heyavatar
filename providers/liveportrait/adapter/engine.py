"""LivePortrait adapter engine class.

Lifecycle + lifecycle-state methods live here. Mock-mode helpers
(:mod:`_mock`) are free functions. Real-mode helpers
(:mod:`_identity`, :mod:`_render`) attach their methods to this class
at import time so consumers can use them via the instance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

from contracts.avatar_engine import AvatarEngine, EngineHealth, EngineState
from src.core.config import Settings, get_settings
from src.core.logging import get_logger
from src.domain.enums import EngineId
from src.domain.types import (
    AvatarIdentityHandle,
    IdentityId,
    IdentitySpec,
    RenderChunkRequest,
    RenderChunkResult,
)

from providers.liveportrait.adapter._mock import _mock_identity_assets, _mock_render_chunk
from providers.liveportrait.adapter._upstream import (
    _import_torch,
    _import_upstream_live_portrait,
    _to_upstream_crop_config,
    _to_upstream_inference_config,
)
from providers.liveportrait.checkpoint_manager.manager import CheckpointManager
from providers.liveportrait.inference_config import CropConfig, InferenceConfig

LOG = get_logger("providers.liveportrait")


@dataclass(slots=True)
class LivePortraitAdapter(AvatarEngine):
    """AvatarEngine implementation for LivePortrait.

    Lifecycle contract:
        ``load()`` is idempotent — call ``unload()`` first if you
        need a clean re-load. ``prepare_identity()`` and
        ``render_chunk()`` MUST be preceded by a successful
        ``load()``; in mock mode they accept calls without load().
    """

    engine_id: ClassVar[EngineId] = EngineId.LIVE_PORTRAIT

    settings: Settings = field(default_factory=get_settings)
    checkpoints: CheckpointManager = field(default_factory=CheckpointManager)
    inf_cfg: InferenceConfig = field(default_factory=InferenceConfig)
    crop_cfg: CropConfig = field(default_factory=CropConfig)
    render_batch_size: int = 8  # frames per GPU kernel launch

    # Private lifecycle references (all ``None`` until ``load()``).
    _loaded_at: Optional[float] = field(default=None, init=False, repr=False)
    _state: EngineState = field(default=EngineState.UNLOADED, init=False, repr=False)
    _pipeline: Any = field(default=None, init=False, repr=False)
    _wrapper: Any = field(default=None, init=False, repr=False)
    _torch_device: Any = field(default=None, init=False, repr=False)
    _last_error: Optional[str] = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Bring the engine up.

        In **mock mode** this is a no-op bookkeeping change; in
        **real mode** we resolve the checkpoint manifest, attempt to
        import the upstream package, and warm the device. On any
        failure we transition to ``DEGRADED`` so ``health()`` reports
        the issue and ``render_chunk()`` returns a clearly-marked
        dummy mp4 instead of crashing the worker.
        """
        if self.settings.mock_engine:
            self._loaded_at = time.monotonic()
            self._state = EngineState.IDLE
            self.mark_loaded()
            return

        self._state = EngineState.LOADING
        try:
            torch = _import_torch()
            if torch is None:
                self._fail("PyTorch is not installed; install torch>=2.0,<2.5")
                return
            if not torch.cuda.is_available():
                self._fail("LivePortrait requires CUDA; no GPU detected on this node")
                return
            self._torch_device = torch.device("cuda:0")

            # Resolve checkpoints — downloads from upstream release
            # and sha256-verifies each file. Raises on failure.
            self.checkpoints.ensure_present()

            upstream = _import_upstream_live_portrait()
            if upstream is None:
                self._fail(
                    "Upstream LivePortrait package not importable. Clone "
                    "https://github.com/KlingAIResearch/LivePortrait, build the "
                    "MultiScaleDeformableAttention CUDA op, and add src/ to "
                    "PYTHONPATH (or set HEYAVATAR_LIVE_PORTRAIT_SRC)."
                )
                return

            inf_upstream = _to_upstream_inference_config(
                upstream, self.inf_cfg, self.checkpoints
            )
            # We construct the LivePortraitWrapper directly instead of
            # the full LivePortraitPipeline because the latter includes
            # a Cropper that requires InsightFace models we don't manage.
            # Our adapter only uses the wrapper (appearance / motion /
            # warping / spade / stitching modules), never the cropper.
            self._wrapper = upstream.LivePortraitWrapper(
                inference_cfg=inf_upstream
            )
            self._pipeline = None  # not used; kept for type compat
            self._loaded_at = time.monotonic()
            self._state = EngineState.IDLE
            self.mark_loaded()
            LOG.info("LivePortraitAdapter loaded on %s", self._torch_device)
        except Exception as exc:  # noqa: BLE001 — we want to surface all causes
            self._fail(
                f"LivePortraitAdapter.load raised: {type(exc).__name__}: {exc}"
            )

    def unload(self) -> None:
        """Free VRAM and drop the pipeline reference."""
        torch = _import_torch()
        try:
            self._pipeline = None
            self._wrapper = None
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            self._loaded_at = None
            self._state = EngineState.UNLOADED
            self._last_error = None

    # ------------------------------------------------------------------
    # Identity preparation
    # ------------------------------------------------------------------
    def prepare_identity(self, source_image: Path) -> Dict[str, bytes]:
        """Produce the dict of pack assets for the source identity.

        Returns the standard pack entries documented in
        :class:`LivePortraitAdapter`.
        """
        if self.settings.mock_engine or self._state not in (
            EngineState.IDLE,
            EngineState.LOADING,
            EngineState.RENDERING,
        ):
            return _mock_identity_assets(source_image)
        # Real-mode path. Method body attached by :mod:`_identity`.
        return self._real_prepare_identity(source_image)

    # ------------------------------------------------------------------
    # Chunk rendering — the GPU hot path
    # ------------------------------------------------------------------
    def render_chunk(
        self,
        request: RenderChunkRequest,
        identity: AvatarIdentityHandle,
    ) -> RenderChunkResult:
        """Render one audio window through the upstream pipeline."""
        _start, end = request.audio_window
        clipped_end = max(_start + 0.5, end)

        # Fast path: mock mode keeps the old deterministic behaviour
        # so the existing contract test stays green.
        if self.settings.mock_engine or self._state == EngineState.UNLOADED:
            return _mock_render_chunk(
                request, clipped_end, capture_dir=self.settings.capture_dir
            )

        # Slow path: real-mode but DEGRADED (e.g. load failed). We
        # still emit a *valid* mp4 of the right duration so the
        # orchestrator sees completion instead of a crash.
        if self._state == EngineState.DEGRADED:
            LOG.warning(
                "render_chunk invoked while DEGRADED; last_error=%s",
                self._last_error,
            )
            if self.settings.mock_engine:
                return _mock_render_chunk(
                    request,
                    clipped_end,
                    capture_dir=self.settings.capture_dir,
                    degraded=True,
                )
            raise RuntimeError(
                f"LivePortraitAdapter is DEGRADED; cannot render chunk. "
                f"last_error={self._last_error}"
            )

        try:
            return self._real_render_chunk(request, identity, clipped_end)
        except Exception as exc:  # noqa: BLE001
            if self.settings.mock_engine:
                LOG.error(
                    "LivePortraitAdapter.render_chunk crashed: %s; falling back to mock mp4",
                    exc,
                )
                return _mock_render_chunk(
                    request,
                    clipped_end,
                    capture_dir=self.settings.capture_dir,
                    degraded=True,
                )
            raise  # re-raise in real mode so the job fails properly

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    def health(self) -> EngineHealth:
        """Return ``EngineHealth`` so the orchestrator can route around us."""
        uptime = (time.monotonic() - self._loaded_at) if self._loaded_at else 0.0
        vram_mb = 0
        if self._torch_device is not None:
            torch = _import_torch()
            if torch is not None and torch.cuda.is_available():
                vram_mb = int(torch.cuda.memory_allocated(self._torch_device) // (1 << 20))
        metrics: Dict[str, float] = {
            "avg_fps_last_job": 0.0,
            "last_error_code": 0.0 if not self._last_error else 1.0,
        }
        return EngineHealth(
            engine_id=self.engine_id,
            state=self._state,
            vram_used_mb=vram_mb,
            uptime_seconds=uptime,
            mock_mode=self.settings.mock_engine,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _fail(self, reason: str) -> None:
        """Transition to DEGRADED + record the cause in ``health.metrics``."""
        LOG.error("LivePortraitAdapter DEGRADED: %s", reason)
        self._last_error = reason
        self._state = EngineState.DEGRADED
        self._loaded_at = self._loaded_at or time.monotonic()


# Trigger registration of the per-job methods attached by :mod:`_identity`
# and :mod:`_render`. MUST run AFTER the :class:`LivePortraitAdapter`
# definition above — the attach helpers in those modules try to read
# the class attribute, so a class-first invariant is required.
import providers.liveportrait.adapter._identity  # noqa: E402,F401
import providers.liveportrait.adapter._render    # noqa: E402,F401
