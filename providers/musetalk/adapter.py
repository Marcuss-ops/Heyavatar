"""MuseTalk adapter — implements :class:`AvatarEngine` for MuseTalk.

MuseTalk (MIT, https://github.com/TMElyralab/MuseTalk) is a real-time
lip-sync model that operates on the 256x256 face region in the VAE latent
space. This adapter copies the LivePortrait shape but uses MuseTalk's
distinct engine id and licensing profile.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict

import numpy as np

from contracts.avatar_engine import EngineHealth, EngineState
from src.core.config import Settings, get_settings
from src.domain.enums import EngineId
from src.domain.types import (
    AvatarIdentityHandle,
    IdentitySpec,
    RenderChunkRequest,
    RenderChunkResult,
)
from providers.liveportrait.adapter import LivePortraitAdapter, _seed_from_path, _write_dummy_mp4


@dataclass(slots=True)
class MuseTalkAdapter(LivePortraitAdapter):
    engine_id: ClassVar[EngineId] = EngineId.MUSE_TALK

    def prepare_identity(self, source_image: Path) -> Dict[str, bytes]:
        if not self.settings.mock_engine:
            raise NotImplementedError("Real-mode prepare_identity lives in a GPU worker image.")
        rng = np.random.default_rng(seed=_seed_from_path(source_image) ^ 0x5A5A)
        return {
            "source_features.bin": rng.standard_normal(64, dtype=np.float32).tobytes(),
            "canonical_keypoints.bin": rng.standard_normal((68, 2), dtype=np.float32).tobytes(),
            "face_mask.png": (rng.random((64, 64)) > 0.5).astype(np.uint8).tobytes(),
            "face_crop.png": (rng.random((128, 128, 3)) * 255).astype(np.uint8).tobytes(),
            "identity_embedding.bin": rng.standard_normal(512, dtype=np.float32).tobytes(),
            "source_latent.bin": rng.standard_normal((4, 16, 16), dtype=np.float32).tobytes(),
        }

    def render_chunk(
        self,
        request: RenderChunkRequest,
        identity: AvatarIdentityHandle,
    ) -> RenderChunkResult:
        if not self.settings.mock_engine:
            raise NotImplementedError("Real-mode render_chunk lives in a GPU worker image.")
        _start, end = request.audio_window
        out_dir = self.settings.capture_dir / request.job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"chunk_{request.chunk_index:04d}.mp4"
        _write_dummy_mp4(out_path, duration=max(0.5, end - _start or 0.5), fps=request.fps)
        # MuseTalk is faster than LivePortrait; report a slightly lower gpu-second.
        return RenderChunkResult(
            chunk_index=request.chunk_index,
            output_path=out_path,
            duration_seconds=max(0.5, end - _start or 0.5),
            frames_rendered=int(max(0.5, end - _start or 0.5) * request.fps),
            gpu_seconds=max(0.005, (end - _start or 0.5) * 0.008),
            engine_id=EngineId.MUSE_TALK,
        )
