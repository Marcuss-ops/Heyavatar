"""LivePortrait adapter — implements :class:`AvatarEngine` for LivePortrait.

LivePortrait (MIT, https://github.com/KlingAIResearch/LivePortrait) is a
keypoint-based portrait animator. There are two modes for this adapter:

* **Real mode** (``HEYAVATAR_MOCK_ENGINE`` unset): the adapter will attempt
  to import the upstream package and run inference. Production deployments
  run real LivePortrait in a dedicated GPU container.
* **Mock mode** (``HEYAVATAR_MOCK_ENGINE=1``): every method returns a
  deterministic synthetic Avatar Pack / chunk .mp4 so the surrounding
  pipeline can be exercised end-to-end without a GPU.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, Tuple

import numpy as np

from contracts.avatar_engine import AvatarEngine, EngineHealth, EngineState
from src.core.config import Settings, get_settings
from src.domain.enums import EngineId
from src.domain.types import (
    AvatarIdentityHandle,
    IdentityId,
    IdentitySpec,
    RenderChunkRequest,
    RenderChunkResult,
)


@dataclass(slots=True)
class LivePortraitAdapter(AvatarEngine):
    engine_id: ClassVar[EngineId] = EngineId.LIVE_PORTRAIT

    settings: Settings = field(default_factory=get_settings)
    _loaded_at: float | None = field(default=None, init=False, repr=False)

    # -- lifecycle -----------------------------------------------------------

    def load(self) -> None:
        if not self.settings.mock_engine:
            raise NotImplementedError(
                "LivePortraitAdapter running in real mode requires the upstream "
                "LivePortrait package installed in this image. Use HEYAVATAR_MOCK_ENGINE=1 "
                "to run without GPU."
            )
        self.mark_loaded()
        self._loaded_at = time.monotonic()

    def unload(self) -> None:
        # In real mode: free keypoint buffers, drop caches.
        # In mock mode: nothing to free other than bookkeeping.
        self._loaded_at = None

    # -- identity prep -------------------------------------------------------

    def prepare_identity(self, source_image: Path) -> Dict[str, bytes]:
        if not self.settings.mock_engine:
            raise NotImplementedError("Real-mode prepare_identity lives in a GPU worker image.")
        # Deterministic synthetic assets used in mock mode. Each entry mirrors
        # the file the real adapter emits, so the pack layout is identical.
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

    # -- chunk rendering -----------------------------------------------------

    def render_chunk(
        self,
        request: RenderChunkRequest,
        identity: AvatarIdentityHandle,
    ) -> RenderChunkResult:
        if not self.settings.mock_engine:
            raise NotImplementedError("Real-mode render_chunk lives in a GPU worker image.")
        _start, end = request.audio_window
        # Mock policy: one synthetic black .mp4 per chunk, written to a
        # deterministic location under capture_dir.
        out_dir = self.settings.capture_dir / request.job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"chunk_{request.chunk_index:04d}.mp4"
        _write_dummy_mp4(out_path, duration=max(0.5, end - _start or 0.5), fps=request.fps)
        return RenderChunkResult(
            chunk_index=request.chunk_index,
            output_path=out_path,
            duration_seconds=max(0.5, end - _start or 0.5),
            frames_rendered=int(max(0.5, end - _start or 0.5) * request.fps),
            gpu_seconds=max(0.01, (end - _start or 0.5) * 0.012),
            engine_id=EngineId.LIVE_PORTRAIT,
        )

    # -- health --------------------------------------------------------------

    def health(self) -> EngineHealth:
        uptime = (time.monotonic() - self._loaded_at) if self._loaded_at else 0.0
        return EngineHealth(
            engine_id=self.engine_id,
            state=EngineState.IDLE if self._loaded_at else EngineState.UNLOADED,
            vram_used_mb=0,
            uptime_seconds=uptime,
            mock_mode=self.settings.mock_engine,
            metrics={"avg_fps_last_job": 0.0},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_from_path(path: Path) -> int:
    return int.from_bytes(path.read_bytes()[:8].ljust(8, b"\0"), "little")


def _write_dummy_mp4(path: Path, *, duration: float, fps: int) -> None:
    """Write a synthetic black .mp4 to ``path`` for a given duration / fps.

    Uses ffmpeg via subprocess when available. If ffmpeg is missing we
    raise a clear ``RuntimeError`` rather than silently writing a 4-byte
    stub that downstream tools (ffprobe, encoding worker's concat
    demuxer) cannot interpret.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "_write_dummy_mp4 requires ffmpeg to produce a valid mp4 stub. "
            "Install it (brew install ffmpeg / apt install ffmpeg) or "
            "run tests in an environment that bundles it."
        )
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
        "-f", "lavfi",
        "-i", f"color=c=0x111111:s=512x512:r={fps}",
        "-t", f"{duration:.3f}",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to produce mock chunk at {path}: "
            f"{result.stderr.strip()}"
        )
