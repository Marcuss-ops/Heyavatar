"""Helpers for sampling small face-motion biases from a timeline.

The goal is not to drive a full facial rig here. The goal is to
translate a lightweight face-motion plan into a few frame-wise bias
signals that the real renderers can consume without changing their
core contracts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.motion.face_motion_timeline import FaceMotionTimeline


@dataclass(slots=True, frozen=True)
class FaceMotionBiasFrame:
    blink: float = 0.0
    brow: float = 0.0
    mouth: float = 0.0
    head: float = 0.0


_MOTION_WEIGHTS: dict[str, FaceMotionBiasFrame] = {
    "face_idle_soft": FaceMotionBiasFrame(),
    "blink_soft": FaceMotionBiasFrame(blink=1.0, brow=0.10),
    "brow_raise_small": FaceMotionBiasFrame(brow=1.0, blink=0.10, mouth=0.05),
    "smile_small": FaceMotionBiasFrame(mouth=1.0, brow=0.20, blink=0.05),
    "nod_small": FaceMotionBiasFrame(head=1.0, mouth=0.08, brow=0.05),
    "question_face": FaceMotionBiasFrame(blink=0.30, brow=1.0, mouth=0.25, head=0.10),
}


def load_face_motion_timeline(path: Path | None) -> FaceMotionTimeline | None:
    if path is None or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return FaceMotionTimeline.from_dict(payload)
    except Exception:
        return None


def sample_face_motion_biases(
    timeline: FaceMotionTimeline | None,
    *,
    frames: int,
    fps: int,
) -> dict[str, list[float]]:
    if timeline is None or frames <= 0 or fps <= 0:
        zero = [0.0] * max(0, frames)
        return {"blink": zero, "brow": zero, "mouth": zero, "head": zero}

    out = {"blink": [0.0] * frames, "brow": [0.0] * frames, "mouth": [0.0] * frames, "head": [0.0] * frames}
    segments = list(timeline.segments)
    if not segments:
        return out

    for idx in range(frames):
        t = min(timeline.duration, max(0.0, (idx + 0.5) / fps))
        seg = None
        for candidate in segments:
            if candidate.start <= t < candidate.end or (
                idx == frames - 1 and candidate.end >= timeline.duration and t >= candidate.start
            ):
                seg = candidate
                break
        if seg is None:
            continue
        weights = _MOTION_WEIGHTS.get(seg.motion_id, FaceMotionBiasFrame())
        scale = max(0.0, float(seg.intensity))
        out["blink"][idx] = min(1.0, weights.blink * scale)
        out["brow"][idx] = min(1.0, weights.brow * scale)
        out["mouth"][idx] = min(1.0, weights.mouth * scale)
        out["head"][idx] = min(1.0, weights.head * scale)

    return out

