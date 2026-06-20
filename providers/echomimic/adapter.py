"""EchoMimic adapter — implements :class:`AvatarEngine` for EchoMimic.

EchoMimic (Apache-2.0, https://github.com/BadToBest/EchoMimic) is a
half-body avatar generator, useful for the Studio tier. Same mock-mode
policy as the other adapters.
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
class EchoMimicAdapter(LivePortraitAdapter):
    engine_id: ClassVar[EngineId] = EngineId.ECHO_MIMIC

    def prepare_identity(self, source_image: Path) -> Dict[str, bytes]:
        if not self.settings.mock_engine:
            raise NotImplementedError("Real-mode prepare_identity lives in a GPU worker image.")
        rng = np.random.default_rng(seed=_seed_from_path(source_image) ^ 0xE0E0)
        # EchoMimic stores half-body crops too.
        return {
            "source_features.bin": rng.standard_normal(128, dtype=np.float32).tobytes(),
            "canonical_keypoints.bin": rng.standard_normal((90, 2), dtype=np.float32).tobytes(),
            "face_mask.png": (rng.random((96, 96)) > 0.5).astype(np.uint8).tobytes(),
            "face_crop.png": (rng.random((192, 192, 3)) * 255).astype(np.uint8).tobytes(),
            "identity_embedding.bin": rng.standard_normal(1024, dtype=np.float32).tobytes(),
            "source_latent.bin": rng.standard_normal((8, 32, 32), dtype=np.float32).tobytes(),
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
        # EchoMimic halves the body; ~50% more time per chunk relative to head-only.
        return RenderChunkResult(
            chunk_index=request.chunk_index,
            output_path=out_path,
            duration_seconds=max(0.5, end - _start or 0.5),
            frames_rendered=int(max(0.5, end - _start or 0.5) * request.fps),
            gpu_seconds=max(0.01, (end - _start or 0.5) * 0.018),
            engine_id=EngineId.ECHO_MIMIC,
        )
