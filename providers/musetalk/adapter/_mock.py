"""Mock-mode helpers for the MuseTalk adapter.

These mirror the contracts of the real-mode methods in
:mod:`_identity` and :mod:`_render` so the mock-mode path is a
drop-in replacement for the real path in tests and CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from providers._ffmpeg import FACE_REGION_RESOLUTION, _seed_from_path, _write_dummy_mp4
from src.domain.types import RenderChunkRequest, RenderChunkResult
from src.domain.enums import EngineId


def _mock_identity_assets(source_image: Path) -> Dict[str, bytes]:
    """Deterministic synthetic pack assets for mock-mode and DEGRADED."""
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
    request: RenderChunkRequest,
    clipped_end: float,
    *,
    capture_dir: Path,
    degraded: bool = False,
) -> RenderChunkResult:
    """Emit a synthetic black mp4 with the right duration."""
    duration = max(0.5, clipped_end - request.audio_window[0])
    out_dir = capture_dir / request.job_id
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
