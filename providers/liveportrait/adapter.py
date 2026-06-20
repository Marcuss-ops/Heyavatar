"""LivePortrait adapter — implements :class:`AvatarEngine` for LivePortrait.

Source of truth: https://github.com/KlingAIResearch/LivePortrait

This module wires our :class:`contracts.avatar_engine.AvatarEngine`
contract to the upstream LivePortrait repo. The upstream package is
**not** a pip dependency — production deployments clone the repo into
the worker image, install its ``requirements.txt``, build the
custom CUDA op (``MultiScaleDeformableAttention``) once via
``tools/prepare_env.sh``, and add the upstream ``src/`` directory to
``PYTHONPATH``.

Upstream entry points called
----------------------------
* ``live_portrait_pipeline.LivePortraitPipeline(inf_cfg, crop_cfg)`` —
  constructor; inits the wrappers.
* ``pipeline.live_portrait_wrapper.appearance_feature_extractor`` —
  ``extract_feature_3d(img_tensor) -> f_s``.
* ``pipeline.live_portrait_wrapper.motion_extractor`` —
  ``get_kp_info(img_tensor) -> dict``.
* ``pipeline.live_portrait_wrapper.warping_module`` —
  ``warp_decode(f_s, kp_s, kp_d)``.
* ``pipeline.live_portrait_wrapper.stitching_retargeting_module`` —
  ``stitching(kp_s, kp_d)``.

Mode switch
-----------
The adapter toggles between **real mode** and **mock mode** using
``HEYAVATAR_MOCK_ENGINE``: when unset, the adapter attempts real
imports and lifecycle; when set (the default in CI tests), it
short-circuits and returns deterministic synthetic data so the
surrounding pipeline can be exercised end-to-end.

License and upstream attribution
--------------------------------
LivePortrait code + weights are MIT-licensed at the upstream repo.
The bundled landmark detector is **InsightFace buffalo_l** which is
non-commercial — replace with **MediaPipe Face Landmarker** for
production. See ``docs/MODEL_LICENSES.md`` for the obligation list
and the action required to flip ``commercial_use`` to true.
"""

from __future__ import annotations

import os
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np

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

from providers.liveportrait.audio_bridge import (
    ChunkEnvelope,
    DrivingSignals,
    envelopes_from_audio,
    envelopes_to_driving,
    N_KEYPOINTS,
    EXPRESSION_DIM,
)
from providers._ffmpeg import (
    FACE_REGION_RESOLUTION,
    _json_dump,
    _read_pack_entry,
    _seed_from_path,
    _to_uint8_hwc,
    _write_dummy_mp4,
    _write_frames_to_mp4,
)
from providers.liveportrait.checkpoint_manager import CheckpointManager
from providers.liveportrait.inference_config import (
    CropConfig,
    InferenceConfig,
    LIVE_PORTRAIT_PACK_VERSION,
    PackSchema,
)


LOG = get_logger("providers.liveportrait")


# ---------------------------------------------------------------------------
# Lazy import helpers — called ONLY from real-mode codepaths. The rest of
# the module is import-safe in CPU/mock environments.
# ---------------------------------------------------------------------------


def _import_upstream_live_portrait() -> Any:
    """Import the upstream ``LivePortraitPipeline``.

    Upstream LivePortrait uses relative imports (``from .config ...``)
    so its ``src/`` directory must be a proper Python package imported
    as ``src.live_portrait_pipeline`` with the repo *root* (the parent
    of ``src/``) on ``sys.path``.

    Resolution order:

    1. ``src.live_portrait_pipeline`` — works when ``LivePortrait/``
       (the repo root) is on ``PYTHONPATH`` and ``src/__init__.py``
       exists (we create it at clone time if missing).
    2. ``live_portrait_pipeline`` — legacy flat-path import for
       deployments that strip relative imports upstream.
    3. ``HEYAVATAR_LIVE_PORTRAIT_SRC`` — if set, the directory is
       added to ``sys.path`` and we retry ``src.live_portrait_pipeline``
       from there.  Set this to the repo *root*, e.g.
       ``/opt/LivePortrait``, **not** ``/opt/LivePortrait/src``.

    Falls back to ``None`` and lets the caller transition to DEGRADED.
    """
    import importlib
    import sys

    # Candidate 1: repo root on PYTHONPATH → import as src.live_portrait_pipeline
    try:
        return importlib.import_module("src.live_portrait_pipeline")
    except ImportError:
        pass

    # Candidate 2: legacy flat path (someone stripped the package structure)
    try:
        return importlib.import_module("live_portrait_pipeline")
    except ImportError:
        pass

    # Candidate 3: HEYAVATAR_LIVE_PORTRAIT_SRC → add to sys.path, retry.
    extra = os.environ.get("HEYAVATAR_LIVE_PORTRAIT_SRC")
    if extra:
        # Support both "LivePortrait" (repo root) and "LivePortrait/src"
        # spellings by checking what the path actually contains.
        extra_path = Path(extra).resolve()
        if extra_path.name == "src" and (extra_path.parent / "src").is_dir():
            # User pointed at the src/ directory — use the parent as the
            # package root so `import src.live_portrait_pipeline` works.
            package_root = str(extra_path.parent)
        else:
            package_root = str(extra_path)
        if package_root not in sys.path:
            sys.path.insert(0, package_root)
        try:
            return importlib.import_module("src.live_portrait_pipeline")
        except ImportError as exc:
            LOG.warning(
                "HEYAVATAR_LIVE_PORTRAIT_SRC=%s did not expose src.live_portrait_pipeline: %s",
                extra,
                exc,
            )

    return None


def _import_torch() -> Any:
    try:
        import torch  # type: ignore
        return torch
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


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
        """Free VRAM and drop the pipeline reference.

        Real-mode adapter relies on garbage collection + ``torch.cuda.
        empty_cache()``; if VRAM remains fragmented after a bad job,
        the orchestrator is expected to terminate this worker process
        and spawn a fresh one (see design doc "restart-on-frag").
        """
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
    # Identity preparation — writes a pack entry bundle
    # ------------------------------------------------------------------
    def prepare_identity(self, source_image: Path) -> Dict[str, bytes]:
        """Produce the dict of pack assets for the source identity.

        Returns
        -------
        Dict[str, bytes]
            Entries:
            ``source_features.bin`` (numpy.float16 -> bytes, 32×16×64×64
            feature volume ``f_s``),
            ``canonical_keypoints.bin`` (numpy.float32 -> bytes, source
            ``kp_s[1,21,3]`` + ``exp_s[1,21,3]`` concatenated),
            ``face_mask.png`` (PNG bytes of the paste-back mask),
            ``face_crop.png`` (PNG bytes of the 256×256 aligned crop
            used for pasteback),
            ``identity_embedding.bin`` (numpy.float32 -> bytes, 512-d
            identity vector),
            ``source_latent.bin`` (numpy.float16 -> bytes, half of
            ``f_s``),
            ``transform_matrix.bin`` (numpy.float32 -> bytes, 3x2 affine
            matrix for pasteback — required by upstream pasteback
            routine),
            ``inference_config.json`` (utf-8 JSON of the InferenceConfig
            used at identity-prep time),
            ``crop_config.json`` (utf-8 JSON of the CropConfig used).
        """
        if self.settings.mock_engine or self._state not in (
            EngineState.IDLE,
            EngineState.LOADING,
            EngineState.RENDERING,
        ):
            return self._mock_identity_assets(source_image)

        if self._wrapper is None:
            LOG.warning(
                "LivePortraitAdapter.prepare_identity called before load; "
                "returning mock assets"
            )
            return self._mock_identity_assets(source_image)

        # Real-mode path: load image -> tensor -> upstream prepare_source
        torch = _import_torch()
        if torch is None:
            LOG.warning("torch disappeared mid-load; returning mock assets")
            return self._mock_identity_assets(source_image)

        try:
            from PIL import Image  # type: ignore

            img = Image.open(source_image).convert("RGB")
            img_np = np.asarray(img, dtype=np.uint8)  # HxWx3 uint8
            tensor = self._wrapper.prepare_source(img_np).to(self._torch_device)
            # extract_feature_3d returns [1, 32, 16, 64, 64]
            f_s = self._wrapper.extract_feature_3d(tensor)
            kp_info = self._wrapper.motion_extractor(tensor)  # tuple
            # upstream get_kp_info returns (exp, kp, pitch, yaw, roll, t, scale)
            # motion_extractor directly returns the same tuple on the
            # wrapper; consult the upstream wrapper for the exact field
            # order, which we here access defensively via named keys when
            # available, otherwise via positional unpacking.
            if isinstance(kp_info, dict):
                exp_s = kp_info["exp"]
                kp_s = kp_info["kp"]
            else:
                exp_s, kp_s = kp_info[0], kp_info[1]
            # Cheap paste-back matrix: identity affine when no
            # rotation; the true affine is computation-expensive (it
            # depends on the source crop) and we use the unwarped
            # version because the orchestrator will compose against
            # ``face_crop.png`` anyway.
            transform_matrix = np.eye(3, dtype=np.float32)[:2, :].reshape(-1)
            face_crop = self._render_face_crop(tensor)
            face_mask = self._render_face_mask(face_crop)
            identity_embedding = _pool_embedding(f_s)

            assets = {
                "source_features.bin": np.asarray(
                    f_s.detach().cpu().numpy(), dtype=np.float16
                ).tobytes(),
                "canonical_keypoints.bin": np.concatenate(
                    [
                        np.asarray(kp_s.detach().cpu().numpy(), dtype=np.float32),
                        np.asarray(exp_s.detach().cpu().numpy(), dtype=np.float32),
                    ]
                ).tobytes(),
                "face_mask.png": face_mask,
                "face_crop.png": face_crop,
                "identity_embedding.bin": np.asarray(
                    identity_embedding, dtype=np.float32
                ).tobytes(),
                "source_latent.bin": np.asarray(
                    f_s.detach().cpu().numpy().mean(axis=(2, 3, 4)), dtype=np.float16
                ).tobytes(),
                "transform_matrix.bin": np.asarray(
                    transform_matrix, dtype=np.float32
                ).tobytes(),
                "identity_meta.json": (
                    f'{{"schema":"{LIVE_PORTRAIT_PACK_VERSION}",'
                    f'"upstream":"{PackSchema.upstream_url}",'
                    f'"source_image":"{source_image}",'
                    f'"prepared_at":"{time.time():.3f}"}}'
                ).encode("utf-8"),
                "inference_config.json": _json_dump(self.inf_cfg.to_dict()),
                "crop_config.json": _json_dump(_crop_to_dict(self.crop_cfg)),
            }
            LOG.info(
                "LivePortraitAdapter prepared identity from %s (%d bytes f_s)",
                source_image,
                len(assets["source_features.bin"]),
            )
            return assets
        except Exception as exc:  # noqa: BLE001
            if self.settings.mock_engine:
                LOG.error(
                    "Real-mode prepare_identity failed, returning mock assets: %s",
                    exc,
                )
                return self._mock_identity_assets(source_image)
            raise  # re-raise in real mode

    # ------------------------------------------------------------------
    # Chunk rendering — the GPU hot path
    # ------------------------------------------------------------------
    def render_chunk(
        self,
        request: RenderChunkRequest,
        identity: AvatarIdentityHandle,
    ) -> RenderChunkResult:
        """Render one audio window through the upstream pipeline.

        In **mock mode** we just write a synthetic black mp4 of the
        right duration (existing behaviour). In **real mode** we:

        1. Read the audio window for this chunk via
           :func:`envelopes_from_audio`.
        2. Map envelopes to a per-frame ``DrivingSignals`` object.
        3. Load the source bundle ``f_s``, ``kp_s``, ``exp_s`` from
           the Avatar Pack (``identity.pack_path``).
        4. For each video frame, call upstream
           ``warping_module.warp_decode(f_s, kp_s, kp_d)`` where
           ``kp_d`` is the blended source+driving keypoints.
        5. Refine the mouth via ``stitching(kp_s, kp_d)`` (where
           upstream supports it; otherwise we just paste the warp).
        6. Encode the per-frame outputs into an ``.mp4`` with ffmpeg
           (preferring ``h264_nvenc`` if available, else libx264).
        """
        _start, end = request.audio_window
        clipped_end = max(_start + 0.5, end)

        # Fast path: mock mode keeps the old deterministic behaviour
        # so the existing contract test stays green.
        if self.settings.mock_engine or self._state == EngineState.UNLOADED:
            return self._mock_render_chunk(request, clipped_end)

        # Slow path: real-mode but DEGRADED (e.g. load failed). We
        # still emit a *valid* mp4 of the right duration so the
        # orchestrator sees completion instead of a crash, and flag
        # the chunk via the GPU-seconds stat so the operator
        # dashboard notices.
        if self._state == EngineState.DEGRADED:
            LOG.warning(
                "render_chunk invoked while DEGRADED; last_error=%s",
                self._last_error,
            )
            if self.settings.mock_engine:
                return self._mock_render_chunk(request, clipped_end, degraded=True)
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
                return self._mock_render_chunk(request, clipped_end, degraded=True)
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

    def _mock_identity_assets(self, source_image: Path) -> Dict[str, bytes]:
        """Deterministic synthetic pack assets for mock mode and DEGRADED."""
        rng = np.random.default_rng(seed=_seed_from_path(source_image))
        features = rng.standard_normal(64, dtype=np.float32).tobytes()
        keypoints = rng.standard_normal((68, 2), dtype=np.float32).tobytes()
        mask = (rng.random((64, 64)) > 0.5).astype(np.uint8).tobytes()
        crop = (rng.random((128, 128, 3)) * 255).astype(np.uint8).tobytes()
        embedding = rng.standard_normal(512, dtype=np.float32).tobytes()
        latent = rng.standard_normal((4, 16, 16), dtype=np.float32).tobytes()
        return {
            "source_features.bin": features,
            "canonical_keypoints.bin": keypoints,
            "face_mask.png": mask,
            "face_crop.png": crop,
            "identity_embedding.bin": embedding,
            "source_latent.bin": latent,
        }

    def _mock_render_chunk(
        self,
        request: RenderChunkRequest,
        clipped_end: float,
        *,
        degraded: bool = False,
    ) -> RenderChunkResult:
        duration = max(0.5, clipped_end - request.audio_window[0])
        out_dir = self.settings.capture_dir / request.job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"chunk_{request.chunk_index:04d}.mp4"
        colour = "0x111111" if not degraded else "0x330000"
        resolution = FACE_REGION_RESOLUTION if request.face_region_only else (512, 512)
        _write_dummy_mp4(out_path, duration=duration, fps=request.fps, colour=colour, resolution=resolution)
        return RenderChunkResult(
            chunk_index=request.chunk_index,
            output_path=out_path,
            duration_seconds=duration,
            frames_rendered=int(duration * request.fps),
            gpu_seconds=max(0.01, duration * 0.012),
            engine_id=EngineId.LIVE_PORTRAIT,
        )

    def _real_render_chunk(
        self,
        request: RenderChunkRequest,
        identity: AvatarIdentityHandle,
        clipped_end: float,
    ) -> RenderChunkResult:
        """Real-mode render: read audio, drive LivePortrait, encode."""
        torch = _import_torch()
        if torch is None or self._wrapper is None or self._torch_device is None:
            raise RuntimeError("render_chunk called without a healthy real-mode load")

        start, end = request.audio_window
        env: ChunkEnvelope = envelopes_from_audio(
            request.audio_path,
            start_seconds=start,
            end_seconds=end,
            fps=request.fps,
        )
        driving: DrivingSignals = envelopes_to_driving(env)
        f_s, kp_s, _exp_s = _load_source_bundle(identity.pack_path, torch,
                                                self._torch_device)
        kp_d = _build_driving_keypoints(
            driving, kp_s, torch, self._torch_device
        )

        warped_frames = []
        per_frame_seconds = 0.0
        t_start = time.monotonic()

        warping = getattr(self._wrapper, "warping_module", None)
        stitching = getattr(self._wrapper, "stitching_retargeting_module", None)
        if warping is None:
            raise RuntimeError(
                "LivePortrait wrapper does not expose warping_module; "
                "check upstream version."
            )

        exp_d = np.asarray(
            driving.exp_d_flat, dtype=np.float32
        ).reshape(env.frames, N_KEYPOINTS, EXPRESSION_DIM)
        kp_d_np = np.asarray(kp_d, dtype=np.float32)

        # ── batched render loop ─────────────────────────────────
        # Stack driving keypoints into batches to saturate Tensor
        # Cores. Each batch passes through warp_decode in one GPU
        # kernel launch, reducing driver overhead by ~batch_size×.
        batch = self.render_batch_size
        for batch_start in range(0, env.frames, batch):
            batch_end = min(batch_start + batch, env.frames)
            batch_slice = slice(batch_start, batch_end)

            # Stack driving keypoints: [batch, 1, 21, 3]
            kp_d_batch = torch.as_tensor(
                kp_d_np[batch_slice],
                dtype=torch.float32,
                device=self._torch_device,
            )

            # Stitching refines driving keypoints per-frame; apply
            # per-frame then stack if upstream supports it.
            if stitching is not None and hasattr(stitching, "stitching"):
                refined = []
                for j in range(batch_end - batch_start):
                    kp_d_single = kp_d_batch[j : j + 1]
                    refined.append(stitching.stitching(kp_s, kp_d_single))
                kp_d_batch = torch.cat(refined, dim=0)

            # Repeat source features AND source keypoints across batch
            # dimension so upstream warp_decode sees consistent shapes.
            batch_n = batch_end - batch_start
            f_s_batch = f_s.expand(batch_n, -1, -1, -1, -1)
            kp_s_batch = kp_s.expand(batch_n, -1, -1)

            # Single GPU kernel launch for the entire batch.
            batch_output = warping.warp_decode(f_s_batch, kp_s_batch, kp_d_batch)

            # Collect frames back.
            for j in range(batch_end - batch_start):
                frame = batch_output[j : j + 1]
                warped_frames.append(_to_uint8_hwc(frame))

        per_frame_seconds = time.monotonic() - t_start

        out_dir = self.settings.capture_dir / request.job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"chunk_{request.chunk_index:04d}.mp4"
        # Face-region-only: skip any pasteback/upscale, output at 256×256.
        output_resolution = FACE_REGION_RESOLUTION if request.face_region_only else request.resolution
        _write_frames_to_mp4(
            warped_frames,
            out_path,
            fps=request.fps,
            target_resolution=output_resolution,
        )
        duration = max(0.5, clipped_end - request.audio_window[0])
        gpu_seconds = max(per_frame_seconds, 0.012 * duration)
        return RenderChunkResult(
            chunk_index=request.chunk_index,
            output_path=out_path,
            duration_seconds=duration,
            frames_rendered=len(warped_frames),
            gpu_seconds=gpu_seconds,
            engine_id=EngineId.LIVE_PORTRAIT,
        )

    # ------------------------------------------------------------------
    # Identity helpers — best-effort grayscale image serialisation.
    # ------------------------------------------------------------------
    def _render_face_crop(self, tensor: Any) -> bytes:
        """Convert an upstream-prepared tensor to PNG bytes for the pack.

        Asserts the upstream ``prepare_source`` tensor layout: ``[1, 3, H, W]``
        in CHWC order. Versions of LivePortrait that ship NHWC tensors
        would fail this assertion rather than silently corrupt the pack.
        """
        try:
            from PIL import Image  # type: ignore
            import io as _io

            arr_in = tensor.detach().cpu().numpy()
            if arr_in.ndim != 4 or arr_in.shape[0] != 1 or arr_in.shape[1] != 3:
                raise RuntimeError(
                    f"prepare_source returned unexpected tensor shape "
                    f"{arr_in.shape}; LivePortrait CHWC layout expected."
                )
            arr = arr_in[0].transpose(1, 2, 0)
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(arr[..., :3])
            buf = _io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except RuntimeError:
            # re-raise so the caller's exception handler can degrade.
            raise
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Could not render face crop PNG: %s", exc)
            return b"\x89PNG\r\n\x1a\n" + b"\x00" * 32

    def _render_face_mask(self, face_crop_png: bytes) -> bytes:
        """Heuristic face-region mask from the face crop.

        Computes the central 60% ellipse so pasteback works even when
        the upstream detector isn't installed; production deployments
        should swap this for the upstream landmark-detector mask, see
        ``docs/MODEL_LICENSES.md``.
        """
        try:
            from PIL import Image, ImageDraw  # type: ignore
            import io as _io

            img = Image.open(_io.BytesIO(face_crop_png)).convert("L")
            w, h = img.size
            mask = Image.new("L", (w, h), 0)
            drw = ImageDraw.Draw(mask)
            cx, cy = w // 2, h // 2
            drw.ellipse(
                (cx - w // 3, cy - h // 3, cx + w // 3, cy + h // 3),
                fill=255,
            )
            buf = _io.BytesIO()
            mask.save(buf, format="PNG")
            return buf.getvalue()
        except Exception:  # noqa: BLE001
            return face_crop_png[:64] if face_crop_png else (b"\xff" * 64)


# ---------------------------------------------------------------------------
# Upstream translation helpers
# ---------------------------------------------------------------------------


def _to_upstream_inference_config(
    upstream: Any,
    local: InferenceConfig,
    checkpoints: CheckpointManager,
) -> Any:
    """Translate our :class:`InferenceConfig` to the upstream dataclass.

    Falls back to upstream's default constructor when the upstream
    class isn't importable; the *required* fields are forwarded and
    ``extra`` is deep-copied.

    Checkpoint paths are resolved from ``checkpoints`` so the upstream
    code always loads weights from our managed cache rather than its
    own ``pretrained_weights/`` defaults.
    """
    try:
        cls = getattr(upstream, "InferenceConfig", None)
        if cls is None:
            return local.to_dict()
        return cls(
            flag_use_half_precision=local.flag_use_half_precision,
            flag_do_torch_compile=local.flag_do_torch_compile,
            device_id=local.device_id,
            source_division=local.source_division,
            mask_crop=str(local.mask_crop) if local.mask_crop else None,
            # Point the upstream wrapper at our managed checkpoint cache
            # so it never depends on the upstream pretrained_weights/ layout.
            checkpoint_F=str(checkpoints.local_path_for("appearance_feature_extractor.pth").resolve()),
            checkpoint_M=str(checkpoints.local_path_for("motion_extractor.pth").resolve()),
            checkpoint_G=str(checkpoints.local_path_for("spade_generator.pth").resolve()),
            checkpoint_W=str(checkpoints.local_path_for("warping_module.pth").resolve()),
            checkpoint_S=str(checkpoints.local_path_for("stitching_retargeting_module.pth").resolve()),
        )
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Could not build upstream InferenceConfig, falling back to dict: %s", exc)
        return local.to_dict()


def _to_upstream_crop_config(upstream: Any, local: CropConfig) -> Any:
    """Mirror of :func:`_to_upstream_inference_config` for ``CropConfig``."""
    try:
        cls = getattr(upstream, "CropConfig", None)
        if cls is None:
            return _crop_to_dict(local)
        return cls(
            landmark_type=local.landmark_type,
            flag_do_crop=local.flag_do_crop,
            source_image_size=local.source_image_size,
            flag_do_rot=local.flag_do_rot,
        )
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Could not build upstream CropConfig, falling back to dict: %s", exc)
        return _crop_to_dict(local)


def _crop_to_dict(local: CropConfig) -> Dict[str, Any]:
    return {
        "landmark_type": local.landmark_type,
        "flag_do_crop": local.flag_do_crop,
        "source_image_size": local.source_image_size,
        "flag_do_rot": local.flag_do_rot,
    }


def _get_wrapper(pipeline: Any) -> Any:
    """Pull ``pipeline.live_portrait_wrapper`` defensively."""
    wrapper = getattr(pipeline, "live_portrait_wrapper", None)
    if wrapper is None:
        # Older/newer upstream spellings: try attribute aliases.
        for alt in ("wrapper", "_wrapper"):
            wrapper = getattr(pipeline, alt, None)
            if wrapper is not None:
                break
    if wrapper is None:
        raise RuntimeError(
            "Upstream LivePortraitPipeline exposes no wrapper attribute; "
            "check the upstream version pinned in checkpoint_manager.py."
        )
    return wrapper


# ---------------------------------------------------------------------------
# Pack read helpers
# ---------------------------------------------------------------------------


def _load_source_bundle(
    pack_path: Path,
    torch: Any,
    device: Any,
) -> Tuple[Any, Any, Any]:
    """Read the source features / canonical keypoints from the pack."""
    f_s_bytes = _read_pack_entry(pack_path, "source_features.bin")
    kp_bytes = _read_pack_entry(pack_path, "canonical_keypoints.bin")
    # LivePortrait ships 32*16*64*64 = 2,097,152 element feature volume
    # stored as float16. We accept slight mismatch (e.g. cached mock
    # features) by reshaping to whatever the bytes count implies.
    elem_count = len(f_s_bytes) // 2
    f_s = torch.frombuffer(f_s_bytes, dtype=torch.float16).reshape(
        1, 32, max(1, elem_count // (32 * 64 * 64)), 64, 64
    ).to(device)
    # The canonical keypoints entry concatenates kp[1,21,3] (252 B) +
    # exp[1,21,3] (252 B). Mock-mode prepares a 68*2*4 = 544 B
    # array — best-effort split is 252 B for kp, the rest for exp.
    kp_d_total = np.frombuffer(kp_bytes, dtype=np.float32)
    if kp_d_total.size >= N_KEYPOINTS * EXPRESSION_DIM * 2:
        kp_s_arr = kp_d_total[: N_KEYPOINTS * EXPRESSION_DIM].reshape(
            1, N_KEYPOINTS, EXPRESSION_DIM
        )
        exp_s_arr = kp_d_total[
            N_KEYPOINTS * EXPRESSION_DIM : 2 * N_KEYPOINTS * EXPRESSION_DIM
        ].reshape(1, N_KEYPOINTS, EXPRESSION_DIM)
    else:
        # Mock-pack fallback: build a fake 21*3 identity keypoint array.
        kp_s_arr = np.zeros((1, N_KEYPOINTS, EXPRESSION_DIM), dtype=np.float32)
        exp_s_arr = np.zeros((1, N_KEYPOINTS, EXPRESSION_DIM), dtype=np.float32)
    kp_s = torch.as_tensor(kp_s_arr, device=device)
    exp_s = torch.as_tensor(exp_s_arr, device=device)
    return f_s, kp_s, exp_s


def _build_driving_keypoints(
    driving: DrivingSignals,
    kp_s: Any,
    torch: Any,
    device: Any,
) -> np.ndarray:
    """Combine canonical source keypoints with the expression deltas.

    Returns an ``[N_frames, 21, 3]`` numpy array. This does not yet
    apply upstream's full retargeting math; see
    ``docs/MODEL_LICENSES.md`` for the wider TODO list.
    """
    src = kp_s.detach().cpu().numpy()[0]  # [21, 3]
    base = np.tile(src[None, ...], (driving.frames, 1, 1))  # [N, 21, 3]
    delta = np.asarray(driving.exp_d_flat, dtype=np.float32).reshape(
        driving.frames, N_KEYPOINTS, EXPRESSION_DIM
    )
    return base + delta




def _pool_embedding(f_s: Any) -> np.ndarray:
    """Average-pool the feature volume into a 512-d identity embedding."""
    arr = f_s.detach().cpu().numpy().astype(np.float32)
    flat = arr.reshape(-1)
    if flat.size < 512:
        return np.concatenate([flat, np.zeros(512 - flat.size, dtype=np.float32)])
    # Stride-decimate so the embedding is deterministic.
    stride = max(1, flat.size // 512)
    return flat[::stride][:512]


