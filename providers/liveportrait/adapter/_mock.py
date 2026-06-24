"""Mock-mode helpers for the LivePortrait adapter.

These mirror the contracts of the real-mode methods in
:mod:`_identity` and :mod:`_render` so the mock-mode path is a
drop-in replacement for the real path in tests and CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from providers._ffmpeg import (
    FACE_REGION_RESOLUTION,
    _seed_from_path,
    _write_dummy_mp4,
    face_motion_signature,
)
from src.domain.enums import EngineId
from src.domain.types import RenderChunkRequest, RenderChunkResult


def _mock_identity_assets(source_image: Path) -> Dict[str, bytes]:
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
    request: RenderChunkRequest,
    clipped_end: float,
    *,
    capture_dir: Path,
    degraded: bool = False,
) -> RenderChunkResult:
    """Emit a synthetic black mp4 of the right duration.

    ``capture_dir`` is passed explicitly so the helper is a pure
    function and the engine class doesn't need to carry it through
    ``self``.
    """
    duration = max(0.5, clipped_end - request.audio_window[0])
    out_dir = capture_dir / request.job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"chunk_{request.chunk_index:04d}.mp4"
    face_profile = face_motion_signature(request.face_motion_timeline_path)
    motion_ids = face_profile.get("motion_ids", [])
    if degraded:
        colour = "0x330000"
    elif "question_face" in motion_ids:
        colour = "0x223344"
    elif "brow_raise_small" in motion_ids:
        colour = "0x334422"
    elif "smile_small" in motion_ids:
        colour = "0x224433"
    else:
        colour = "0x111111"
    resolution = FACE_REGION_RESOLUTION if request.face_region_only else (512, 512)
    _write_dummy_mp4(out_path, duration=duration, fps=request.fps, colour=colour, resolution=resolution)
    sidecar = out_path.with_suffix(".face_motion.json")
    sidecar.write_text(
        __import__("json").dumps(
            {
                "engine_id": EngineId.LIVE_PORTRAIT.value,
                "degraded": degraded,
                "colour": colour,
                "face_motion": face_profile,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return RenderChunkResult(
        chunk_index=request.chunk_index,
        output_path=out_path,
        duration_seconds=duration,
        frames_rendered=int(duration * request.fps),
        gpu_seconds=max(0.01, duration * 0.012),
        engine_id=EngineId.LIVE_PORTRAIT,
    )
