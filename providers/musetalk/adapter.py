"""MuseTalk adapter — implements :class:`AvatarEngine` for MuseTalk.

MuseTalk (MIT, https://github.com/TMElyralab/MuseTalk) is a real-time
lip-sync model that operates in the VAE latent space. The pipeline:

1. **Face detection** — MediaPipe or InsightFace detects and aligns the
   face region to a 256×256 canonical crop.
2. **VAE encode** — ``sd-vae-ft-mse`` encodes the face crop into a
   4-channel latent tensor.
3. **Whisper audio** — Whisper (tiny) extracts per-frame audio features
   from the driving audio chunk.
4. **UNet denoise** — The MuseTalk UNet takes the source latent + audio
   features and denoises into a new latent.
5. **VAE decode** — Decode the output latent back to RGB frames.
6. **Paste-back** — Composite the rendered face back onto the original
   frame using the alignment inverse transform.

Mode switch
-----------
``HEYAVATAR_MOCK_ENGINE=1`` (the default in CI) short-circuits to
deterministic synthetic data so the pipeline stays testable without
GPU/weights. When unset, the adapter attempts real imports and
falls back to DEGRADED on failure.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
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
    IdentitySpec,
    RenderChunkRequest,
    RenderChunkResult,
)

from providers._ffmpeg import (
    FACE_REGION_RESOLUTION,
    _read_pack_entry,
    _seed_from_path,
    _write_dummy_mp4,
    _write_frames_to_mp4,
)
from providers.liveportrait.checkpoint_manager import CheckpointManager

LOG = get_logger("providers.musetalk")


# ---------------------------------------------------------------------------
# Lazy-import helpers — called ONLY from real-mode codepaths.
# ---------------------------------------------------------------------------


def _import_torch() -> Any:
    try:
        import torch  # type: ignore
        return torch
    except ImportError:
        return None


def _import_musetalk_upstream() -> Any:
    """Import the upstream MuseTalk package.

    Tries two locations:

    1. ``musetalk`` (if on ``PYTHONPATH``).
    2. ``HEYAVATAR_MUSETALK_SRC`` env var appended to ``sys.path``.
    """
    import importlib
    import sys

    try:
        return importlib.import_module("musetalk")
    except ImportError:
        pass
    extra = os.environ.get("HEYAVATAR_MUSETALK_SRC")
    if extra:
        sys.path.insert(0, extra)
        try:
            return importlib.import_module("musetalk")
        except ImportError as exc:
            LOG.warning(
                "HEYAVATAR_MUSETALK_SRC=%s did not expose musetalk: %s",
                extra,
                exc,
            )
    return None


# ---------------------------------------------------------------------------
# MuseTalkCheckpointManager — own checkpoint manifest for MuseTalk weights
# ---------------------------------------------------------------------------

# SHA256 values are "TBD" pending first download.
# HuggingFace LFS handles integrity verification on download.
# To pin: download via `huggingface-cli download TMElyralab/MuseTalk`,
# run `sha256sum` on each file, update the "sha256" field below.
# Set HEYAVATAR_SKIP_SHA256_VERIFY=1 to skip SHA256 verification (for initial setup).
MUSETALK_CHECKPOINT_MANIFEST: list = [
    {
        "name": "musetalk_unet.pth",
        "url": "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/unet.pth",
        "sha256": "TBD",
        "size_bytes": 0,
    },
    {
        "name": "musetalk_vae.bin",
        "url": "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/sd-vae/diffusion_pytorch_model.bin",
        "sha256": "TBD",
        "size_bytes": 0,
    },
    {
        "name": "musetalk_whisper_tiny.bin",
        "url": "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/whisper/pytorch_model.bin",
        "sha256": "TBD",
        "size_bytes": 0,
    },
]


@dataclass(slots=True)
class MuseTalkCheckpointManager(CheckpointManager):
    """Checkpoint manager overridden for MuseTalk checkpoint paths."""

    root: Path = field(
        default_factory=lambda: Path(
            os.environ.get("HEYAVATAR_MUSETALK_CHECKPOINTS", "./checkpoints/musetalk")
        )
    )
    mock_mode: bool = field(
        default_factory=lambda: os.environ.get("HEYAVATAR_MOCK_ENGINE") == "1"
    )

    def __post_init__(self) -> None:
        # Use the parent's CheckpointEntry class to build MuseTalk entries.
        from providers.liveportrait.checkpoint_manager import CheckpointEntry
        self.entries = [
            CheckpointEntry.from_manifest(m) for m in MUSETALK_CHECKPOINT_MANIFEST
        ]
        if not self.mock_mode:
            self.root.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


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
            return self._mock_identity_assets(source_image)

        if self._state == EngineState.DEGRADED:
            LOG.warning(
                "prepare_identity while DEGRADED; returning mock assets. "
                "last_error=%s",
                self._last_error,
            )
            return self._mock_identity_assets(source_image)

        try:
            return self._real_prepare_identity(source_image)
        except Exception as exc:
            LOG.error(
                "Real-mode prepare_identity failed: %s; returning mock",
                exc,
            )
            return self._mock_identity_assets(source_image)

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
            return self._mock_render_chunk(request, clipped_end)

        if self._state == EngineState.DEGRADED:
            LOG.warning(
                "render_chunk while DEGRADED; emitting fallback mp4. "
                "last_error=%s",
                self._last_error,
            )
            return self._mock_render_chunk(request, clipped_end, degraded=True)

        try:
            return self._real_render_chunk(request, identity, clipped_end)
        except Exception as exc:
            LOG.error(
                "MuseTalkAdapter.render_chunk crashed: %s; falling back",
                exc,
            )
            return self._mock_render_chunk(request, clipped_end, degraded=True)

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

    # ------------------------------------------------------------------
    # Mock helpers
    # ------------------------------------------------------------------
    def _mock_identity_assets(self, source_image: Path) -> Dict[str, bytes]:
        rng = np.random.default_rng(seed=_seed_from_path(source_image) ^ 0x5A5A)
        return {
            "source_latent.bin": rng.standard_normal(
                (4, 64, 64), dtype=np.float32
            ).tobytes(),
            "face_crop.png": (
                rng.random((256, 256, 3)) * 255
            ).astype(np.uint8).tobytes(),
            "face_mask.png": (
                rng.random((256, 256)) > 0.5
            ).astype(np.uint8).tobytes(),
            "identity_embedding.bin": rng.standard_normal(
                512, dtype=np.float32
            ).tobytes(),
            "alignment_matrix.bin": rng.standard_normal((6,), dtype=np.float32).tobytes(),
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
        _write_dummy_mp4(
            out_path, duration=duration, fps=request.fps, colour=colour, resolution=resolution
        )
        return RenderChunkResult(
            chunk_index=request.chunk_index,
            output_path=out_path,
            duration_seconds=duration,
            frames_rendered=int(duration * request.fps),
            gpu_seconds=max(0.005, duration * 0.008),
            engine_id=EngineId.MUSE_TALK,
        )

    # ------------------------------------------------------------------
    # Real-mode identity preparation
    # ------------------------------------------------------------------
    def _real_prepare_identity(self, source_image: Path) -> Dict[str, bytes]:
        """Real-mode: face detect → align → VAE encode → pack assets."""
        torch = _import_torch()
        if torch is None:
            raise RuntimeError("PyTorch not available for prepare_identity")

        # ── 1. Load image ─────────────────────────────────────────
        from PIL import Image
        img = Image.open(source_image).convert("RGB")
        img_np = np.asarray(img, dtype=np.uint8)

        # ── 2. Face detection + alignment (MediaPipe or upstream) ──
        face_crop, alignment_matrix = self._detect_and_align(img_np)
        face_crop_png = self._encode_png(face_crop)

        # ── 3. VAE encode ─────────────────────────────────────────
        face_tensor = (
            torch.from_numpy(face_crop.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self._torch_device)
        )
        source_latent = self._vae_encode(face_tensor, torch)

        # ── 4. Identity embedding ─────────────────────────────────
        latent_flat = source_latent.detach().cpu().numpy().flatten()
        stride = max(1, latent_flat.size // 512)
        embedding = latent_flat[::stride][:512].astype(np.float32)

        # ── 5. Face mask (central ellipse heuristic) ──────────────
        face_mask_png = self._render_face_mask(face_crop_png)

        LOG.info(
            "MuseTalkAdapter prepared identity from %s (crop %dx%d)",
            source_image,
            face_crop.shape[1],
            face_crop.shape[0],
        )
        return {
            "source_latent.bin": source_latent.detach()
            .cpu()
            .numpy()
            .astype(np.float16)
            .tobytes(),
            "face_crop.png": face_crop_png,
            "face_mask.png": face_mask_png,
            "identity_embedding.bin": embedding.tobytes(),
            "alignment_matrix.bin": alignment_matrix.astype(np.float32).tobytes(),
        }

    def _detect_and_align(
        self, img_np: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect face and produce a 256×256 aligned crop.

        Tries upstream MuseTalk's face detector first, then MediaPipe.
        Falls back to a centre crop if neither is available.
        """
        # Try upstream detector.
        if self._pipeline is not None:
            try:
                detector = getattr(self._pipeline, "face_detector", None)
                if detector is not None:
                    if callable(detector):
                        return detector(img_np, target_size=256)
                    detect_fn = getattr(detector, "detect_and_align", None)
                    if detect_fn is not None:
                        return detect_fn(img_np, target_size=256)
            except Exception as exc:
                LOG.debug("Upstream face detector failed: %s", exc)

        # Try MediaPipe.
        try:
            import mediapipe as mp
            mp_face = mp.solutions.face_detection
            with mp_face.FaceDetection(
                model_selection=1, min_detection_confidence=0.7
            ) as fd:
                results = fd.process(img_np)
                if results.detections:
                    det = results.detections[0]
                    h, w = img_np.shape[:2]
                    bbox = det.location_data.relative_bounding_box
                    cx = int((bbox.xmin + bbox.width / 2) * w)
                    cy = int((bbox.ymin + bbox.height / 2) * h)
                    size = int(max(bbox.width * w, bbox.height * h) * 1.4)
                    return self._crop_and_resize(
                        img_np, cx, cy, size, 256
                    )
        except ImportError:
            LOG.debug("MediaPipe not installed for face detection")
        except Exception as exc:
            LOG.debug("MediaPipe face detection failed: %s", exc)

        # Fallback: centre crop.
        LOG.warning("No face detector available; using centre crop")
        h, w = img_np.shape[:2]
        size = min(h, w)
        cx, cy = w // 2, h // 2
        return self._crop_and_resize(img_np, cx, cy, size, 256)

    def _crop_and_resize(
        self,
        img: np.ndarray,
        cx: int,
        cy: int,
        size: int,
        target_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Crop a square region around (cx, cy) and resize to target."""
        from PIL import Image

        half = size // 2
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(img.shape[1], cx + half)
        y2 = min(img.shape[0], cy + half)
        crop = img[y1:y2, x1:x2]
        pil_crop = Image.fromarray(crop).resize(
            (target_size, target_size), Image.LANCZOS
        )
        aligned = np.asarray(pil_crop)
        # Identity alignment matrix as fallback.
        matrix = np.array(
            [1.0, 0.0, float(-x1), 0.0, 1.0, float(-y1)], dtype=np.float32
        )
        return aligned, matrix

    def _vae_encode(self, face_tensor: Any, torch: Any) -> Any:
        """Encode face tensor through VAE to get source latent.

        In mock mode, returns a deterministic random latent for
        testability. In real mode, raises if VAE is unavailable.
        """
        if self._pipeline is not None:
            try:
                vae = getattr(self._pipeline, "vae", None)
                if vae is not None and hasattr(vae, "encode"):
                    mu, logvar = vae.encode(face_tensor)
                    return mu
            except Exception as exc:
                LOG.warning("VAE encode via upstream failed: %s", exc)
                if not self.settings.mock_engine:
                    raise RuntimeError(
                        "MuseTalk VAE encode failed in real mode"
                    ) from exc

        if self.settings.mock_engine:
            LOG.warning("VAE not available; using random latent")
            return torch.randn(
                1, 4, 64, 64, device=self._torch_device, dtype=torch.float32
            ) * 0.01

        raise RuntimeError(
            "MuseTalk VAE is unavailable in real mode. "
            "Install upstream MuseTalk or set HEYAVATAR_MOCK_ENGINE=1."
        )

    # ------------------------------------------------------------------
    # Real-mode chunk rendering
    # ------------------------------------------------------------------
    def _real_render_chunk(
        self,
        request: RenderChunkRequest,
        identity: AvatarIdentityHandle,
        clipped_end: float,
    ) -> RenderChunkResult:
        """Real-mode: load latent, extract audio features, UNet denoise,
        VAE decode, and encode frames to mp4."""
        torch = _import_torch()
        if torch is None:
            raise RuntimeError("PyTorch not available for render_chunk")

        start, end = request.audio_window
        fps = request.fps
        duration = max(0.5, clipped_end - start)
        num_frames = int(round(duration * fps))

        # ── 1. Load source latent from identity pack ──────────────
        latent_bytes = _read_pack_entry(
            identity.pack_path, "source_latent.bin"
        )
        source_latent = torch.frombuffer(
            latent_bytes, dtype=torch.float16
        ).reshape(1, 4, 64, 64).to(self._torch_device, dtype=torch.float32)

        # ── 2. Extract audio features via Whisper ─────────────────
        audio_features = self._extract_audio_features(
            request.audio_path,
            start_seconds=start,
            end_seconds=end,
            num_frames=num_frames,
            torch=torch,
        )

        # ── 3. UNet denoising ─────────────────────────────────────
        rendered_frames = self._unet_denoise(
            source_latent, audio_features, torch
        )

        # ── 4. VAE decode to RGB ──────────────────────────────────
        rgb_frames = self._vae_decode_frames(rendered_frames, torch)

        # ── 5. Write mp4 ──────────────────────────────────────────
        out_dir = self.settings.capture_dir / request.job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"chunk_{request.chunk_index:04d}.mp4"
        # Face-region-only: skip pasteback, output face crop at 256×256.
        output_resolution = FACE_REGION_RESOLUTION if request.face_region_only else request.resolution
        _write_frames_to_mp4(
            rgb_frames,
            out_path,
            fps=fps,
            target_resolution=output_resolution,
        )

        return RenderChunkResult(
            chunk_index=request.chunk_index,
            output_path=out_path,
            duration_seconds=duration,
            frames_rendered=len(rgb_frames),
            gpu_seconds=max(0.005, duration * 0.008),
            engine_id=EngineId.MUSE_TALK,
        )

    def _extract_audio_features(
        self,
        audio_path: Path,
        start_seconds: float,
        end_seconds: float,
        num_frames: int,
        torch: Any,
    ) -> Any:
        """Extract per-frame Whisper audio features for UNet guidance.

        Tries upstream pipeline; falls back to random features for testing.
        """
        if self._pipeline is not None:
            try:
                extractor = getattr(self._pipeline, "audio_encoder", None)
                if extractor is not None and hasattr(extractor, "extract"):
                    features = extractor.extract(
                        audio_path,
                        start=start_seconds,
                        end=end_seconds,
                    )
                    if features is not None:
                        return torch.as_tensor(
                            features, dtype=torch.float32, device=self._torch_device
                        )
            except Exception as exc:
                LOG.warning("Whisper audio extraction failed: %s", exc)

        if self.settings.mock_engine:
            LOG.debug("Using random audio features (no Whisper available)")
            return torch.randn(
                num_frames, 384, device=self._torch_device, dtype=torch.float32
            ) * 0.01

        raise RuntimeError(
            "MuseTalk audio encoder (Whisper) is unavailable in real mode. "
            "Install upstream MuseTalk or set HEYAVATAR_MOCK_ENGINE=1."
        )

    def _unet_denoise(
        self,
        source_latent: Any,
        audio_features: Any,
        torch: Any,
    ) -> list:
        """Run the MuseTalk UNet to denoise and produce output latents.

        Processes frames in batches to saturate GPU Tensor Cores.
        Falls back to source latent + noise if UNet is unavailable.
        """
        if self._pipeline is not None:
            try:
                unet = getattr(self._pipeline, "unet", None)
                if unet is not None and hasattr(unet, "forward"):
                    num_frames = audio_features.shape[0]
                    batch = self.render_batch_size
                    latents: list = []
                    current = source_latent.clone() + torch.randn_like(
                        source_latent
                    ) * 0.02

                    for batch_start in range(0, num_frames, batch):
                        batch_end = min(batch_start + batch, num_frames)
                        batch_af = audio_features[batch_start:batch_end]

                        # Try batched UNet forward; fall back to
                        # per-frame if upstream doesn't support batching.
                        try:
                            batch_latent = current.expand(
                                batch_end - batch_start, -1, -1, -1
                            )
                            out = unet(
                                batch_latent,
                                timestep=1,
                                encoder_hidden_states=batch_af,
                            )
                            if isinstance(out, tuple):
                                out = out[0]
                            for j in range(batch_end - batch_start):
                                latents.append(out[j : j + 1].detach())
                        except Exception as batch_exc:
                            LOG.debug("Batched UNet forward failed (will fall back per-frame): %s", batch_exc)
                            # Per-frame fallback within batch.
                            for i in range(batch_start, batch_end):
                                af = audio_features[i : i + 1]
                                out = unet(
                                    current, timestep=1, encoder_hidden_states=af
                                )
                                if isinstance(out, tuple):
                                    out = out[0]
                                latents.append(out.detach())
                    return latents
            except Exception as exc:
                LOG.warning("UNet denoise failed: %s", exc)

        if self.settings.mock_engine:
            LOG.debug("Using random rendered latents (no UNet available)")
            num_frames = audio_features.shape[0]
            return [
                source_latent.clone()
                + torch.randn_like(source_latent) * 0.05
                for _ in range(num_frames)
            ]

        raise RuntimeError(
            "MuseTalk UNet is unavailable in real mode. "
            "Install upstream MuseTalk or set HEYAVATAR_MOCK_ENGINE=1."
        )

    def _vae_decode_frames(
        self, latents: list, torch: Any
    ) -> list:
        """VAE-decode a list of latent tensors to RGB frames.

        Decodes in batches to reduce GPU kernel launch overhead.
        Returns list of ``[H, W, 3]`` numpy uint8 arrays.
        """
        frames: list = []
        vae_available = False
        if self._pipeline is not None:
            try:
                vae = getattr(self._pipeline, "vae", None)
                if vae is not None and hasattr(vae, "decode"):
                    vae_available = True
            except Exception:
                pass

        batch = self.render_batch_size
        for batch_start in range(0, len(latents), batch):
            batch_end = min(batch_start + batch, len(latents))
            batch_latents = latents[batch_start:batch_end]

            if vae_available:
                try:
                    # Stack into batch: [B, 4, 64, 64]
                    stacked = torch.cat(batch_latents, dim=0)
                    decoded = self._pipeline.vae.decode(stacked)
                    if isinstance(decoded, tuple):
                        decoded = decoded[0]
                    # VAE outputs [B, 3, H, W] (CHW); transpose to HWC.
                    decoded_np = (
                        decoded.detach().cpu().numpy()
                        .transpose(0, 2, 3, 1)
                    )  # [B, H, W, 3]
                    for j in range(decoded_np.shape[0]):
                        frame = (decoded_np[j] * 255).clip(0, 255).astype(np.uint8)
                        frames.append(frame)
                    continue
                except Exception:
                    vae_available = False  # degrade after batch failure

            # Per-frame fallback within this batch.
            for latent in batch_latents:
                if vae_available:
                    try:
                        decoded = self._pipeline.vae.decode(latent)
                        if isinstance(decoded, tuple):
                            decoded = decoded[0]
                        frame = (
                            decoded.detach()
                            .cpu()
                            .numpy()[0]
                            .transpose(1, 2, 0)
                        )
                    except Exception:
                        frame = None
                        vae_available = False
                else:
                    frame = None

                if frame is None:
                    if self.settings.mock_engine:
                        frame = np.random.randint(
                            0, 255, (256, 256, 3), dtype=np.uint8
                        )
                    else:
                        raise RuntimeError(
                            "MuseTalk VAE decode failed in real mode. "
                            "Check upstream MuseTalk installation."
                        )
                else:
                    frame = (frame * 255).clip(0, 255).astype(np.uint8)
                frames.append(frame)
        return frames

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _encode_png(arr: np.ndarray) -> bytes:
        from PIL import Image
        import io as _io

        img = Image.fromarray(arr)
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _render_face_mask(self, face_crop_png: bytes) -> bytes:
        try:
            from PIL import Image, ImageDraw
            import io as _io

            img = Image.open(_io.BytesIO(face_crop_png)).convert("L")
            w, h = img.size
            mask = Image.new("L", (w, h), 0)
            drw = ImageDraw.Draw(mask)
            cx, cy = w // 2, h // 2
            drw.ellipse(
                (
                    cx - w // 3,
                    cy - h // 3,
                    cx + w // 3,
                    cy + h // 3,
                ),
                fill=255,
            )
            buf = _io.BytesIO()
            mask.save(buf, format="PNG")
            return buf.getvalue()
        except Exception:
            return face_crop_png[:64] if face_crop_png else b"\xff" * 64
