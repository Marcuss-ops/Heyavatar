"""MuseTalk adapter engine class.

Lifecycle + lifecycle-state methods live here. Mock-mode and real-mode
helpers (:mod:`_mock`, :mod:`_identity`, :mod:`_render`) attach their
methods on this class at import time so callers can use them through
the instance.
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
    IdentitySpec,
    RenderChunkRequest,
    RenderChunkResult,
)

from providers.musetalk.adapter._mock import _mock_identity_assets, _mock_render_chunk
from providers.musetalk.adapter._upstream import _import_torch
from providers.musetalk.adapter.checkpoints import MuseTalkCheckpointManager

LOG = get_logger("providers.musetalk")


@dataclass(slots=True)
class MuseTalkAdapter(AvatarEngine):
    """AvatarEngine implementation for MuseTalk.

    Lifecycle:
        ``load()`` → ``prepare_identity()`` → ``render_chunk()``
    """

    engine_id: ClassVar[EngineId] = EngineId.MUSE_TALK

    # MuseTalk's pack stores VAE latent + face assets, not LivePortrait's
    # source_features / canonical_keypoints.
    pack_required_entries: ClassVar[tuple] = (
        "manifest.json",
        "source_latent.bin",
        "face_mask.png",
        "face_crop.png",
        "identity_embedding.bin",
    )

    settings: Settings = field(default_factory=get_settings)
    checkpoints: MuseTalkCheckpointManager = field(
        default_factory=MuseTalkCheckpointManager
    )
    render_batch_size: int = 8  # frames per GPU kernel launch

    # Private lifecycle references.
    _loaded_at: Optional[float] = field(default=None, init=False, repr=False)
    _state: EngineState = field(default=EngineState.UNLOADED, init=False, repr=False)
    _torch_device: Any = field(default=None, init=False, repr=False)
    _pipeline: Any = field(default=None, init=False, repr=False)
    _last_error: Optional[str] = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Bring the engine up, resolving checkpoints and upstream imports.

        In mock mode this is a no-op. In real mode we import PyTorch,
        verify CUDA, resolve checkpoints, and import upstream MuseTalk.
        On failure we transition to DEGRADED.
        """
        if self.settings.mock_engine:
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
                self._fail("MuseTalk requires CUDA; no GPU detected on this node")
                return
            self._torch_device = torch.device("cuda:0")

            self.checkpoints.ensure_present()

            from providers.musetalk.adapter._upstream import _import_musetalk_upstream
            upstream = _import_musetalk_upstream()
            if upstream is None:
                self._fail(
                    "Upstream MuseTalk package not importable. Clone "
                    "https://github.com/TMElyralab/MuseTalk and set "
                    "HEYAVATAR_MUSETALK_SRC or add to PYTHONPATH."
                )
                return

            self._pipeline = upstream
            self._state = EngineState.IDLE
            self.mark_loaded()
            LOG.info("MuseTalkAdapter loaded on %s", self._torch_device)
        except Exception as exc:
            self._fail(
                f"MuseTalkAdapter.load raised: {type(exc).__name__}: {exc}"
            )

    def unload(self) -> None:
        """Free VRAM and drop pipeline references."""
        torch = _import_torch()
        try:
            self._pipeline = None
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            self._state = EngineState.UNLOADED
            self._last_error = None

    # ------------------------------------------------------------------
    # Identity preparation
    # ------------------------------------------------------------------
    def prepare_identity(self, source_image: Path) -> Dict[str, bytes]:
        """Run face detection + VAE encode to produce identity assets.

        Returns a dict of pack entries:

        * ``source_latent.bin`` — VAE-encoded face latent (float16).
        * ``face_crop.png`` — 256×256 aligned face crop (PNG).
        * ``face_mask.png`` — paste-back mask (PNG).
        * ``identity_embedding.bin`` — pooled embedding (float32).
        * ``alignment_matrix.bin`` — affine alignment matrix (float32).
        """
        if self.settings.mock_engine or self._state == EngineState.UNLOADED:
            return _mock_identity_assets(source_image)

        if self._state == EngineState.DEGRADED:
            LOG.warning(
                "prepare_identity while DEGRADED; returning mock assets. "
                "last_error=%s",
                self._last_error,
            )
            return _mock_identity_assets(source_image)

        try:
            return self._real_prepare_identity(source_image)
        except Exception as exc:
            LOG.error(
                "Real-mode prepare_identity failed: %s; returning mock",
                exc,
            )
            return _mock_identity_assets(source_image)

    # ------------------------------------------------------------------
    # Chunk rendering
    # ------------------------------------------------------------------
    def render_chunk(
        self,
        request: RenderChunkRequest,
        identity: AvatarIdentityHandle,
    ) -> RenderChunkResult:
        """Render one audio window via Whisper → UNet → VAE.

        Mock mode: synthetic black mp4 of the right duration.
        Real mode: loads source latent from pack, extracts audio features
        via Whisper, runs UNet denoising, VAE decodes, and writes mp4.
        """
        _start, end = request.audio_window
        clipped_end = max(_start + 0.5, end)

        if self.settings.mock_engine or self._state == EngineState.UNLOADED:
            return _mock_render_chunk(
                request, clipped_end, capture_dir=self.settings.capture_dir
            )

        if self._state == EngineState.DEGRADED:
            LOG.warning(
                "render_chunk while DEGRADED; emitting fallback mp4. "
                "last_error=%s",
                self._last_error,
            )
            return _mock_render_chunk(
                request,
                clipped_end,
                capture_dir=self.settings.capture_dir,
                degraded=True,
            )

        try:
            return self._real_render_chunk(request, identity, clipped_end)
        except Exception as exc:
            LOG.error(
                "MuseTalkAdapter.render_chunk crashed: %s; falling back",
                exc,
            )
            return _mock_render_chunk(
                request,
                clipped_end,
                capture_dir=self.settings.capture_dir,
                degraded=True,
            )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    def health(self) -> EngineHealth:
        uptime = (time.monotonic() - self._loaded_at) if self._loaded_at else 0.0
        vram_mb = 0
        if self._torch_device is not None:
            torch = _import_torch()
            if torch is not None and torch.cuda.is_available():
                vram_mb = int(
                    torch.cuda.memory_allocated(self._torch_device) // (1 << 20)
                )
        return EngineHealth(
            engine_id=self.engine_id,
            state=self._state,
            vram_used_mb=vram_mb,
            uptime_seconds=uptime,
            mock_mode=self.settings.mock_engine,
            metrics={
                "last_error_code": 0.0 if not self._last_error else 1.0,
            },
        )

    # ------------------------------------------------------------------
    # Internal: failure
    # ------------------------------------------------------------------
    def _fail(self, reason: str) -> None:
        LOG.error("MuseTalkAdapter DEGRADED: %s", reason)
        self._last_error = reason
        self._state = EngineState.DEGRADED
        self._loaded_at = self._loaded_at or time.monotonic()


# Trigger registration of the per-job methods attached by :mod:`_identity`
# and :mod:`_render`. MUST run AFTER the :class:`MuseTalkAdapter`
# definition above — the attach helpers in those modules try to read
# the class attribute, so a class-first invariant is required.
import providers.musetalk.adapter._identity  # noqa: E402,F401
import providers.musetalk.adapter._render    # noqa: E402,F401
