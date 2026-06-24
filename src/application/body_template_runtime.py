"""Runtime helpers for body templates.

The prerecorded body template assets are usually short. For speaking
demos we stretch or trim them to match the target audio duration so the
final muxed video stays duration-aligned.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.domain.body_template import BodyTemplate


def match_body_template_duration(
    body: BodyTemplate,
    target_seconds: float,
    *,
    output_dir: Path,
) -> BodyTemplate:
    """Return a body template whose media duration matches ``target_seconds``.

    If the underlying template already matches the requested length
    within one frame, the original paths are returned unchanged.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    source_cap = cv2.VideoCapture(str(body.body_video))
    if not source_cap.isOpened():
        raise RuntimeError(f"Could not open body video: {body.body_video}")

    fps = float(source_cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_count = int(source_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_cap.release()

    target_frames = max(1, int(round(target_seconds * fps)))
    if frame_count > 0 and abs(target_frames - frame_count) <= 1:
        return body

    return BodyTemplate(
        body_video=_rewrite_video(body.body_video, output_dir / "body.mp4", target_frames=target_frames),
        face_mask=_rewrite_video(body.face_mask, output_dir / "face_mask.mp4", target_frames=target_frames),
        neck_mask=_rewrite_video(body.neck_mask, output_dir / "neck_mask.mp4", target_frames=target_frames),
        face_transforms=_rewrite_transforms(
            body.face_transforms, output_dir / "face_transforms.npz", target_frames=target_frames
        ),
        metadata=_copy_metadata(body.metadata, output_dir / "metadata.json"),
    )


def _rewrite_video(source: Path, dest: Path, *, target_frames: int) -> Path:
    frames, fps, size = _read_video_frames(source)
    if not frames:
        raise RuntimeError(f"Body template video has no frames: {source}")
    if len(frames) == target_frames:
        return source

    dest.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dest), fourcc, fps, size)
    try:
        for idx in range(target_frames):
            writer.write(frames[idx % len(frames)])
    finally:
        writer.release()
    return dest


def _read_video_frames(path: Path) -> tuple[list[np.ndarray], float, tuple[int, int]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open body video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frames: list[np.ndarray] = []
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if size[0] == 0 or size[1] == 0:
            size = (frame.shape[1], frame.shape[0])
        frames.append(frame)
    cap.release()
    return frames, fps, size


def _rewrite_transforms(source: Path, dest: Path, *, target_frames: int) -> Path:
    data = np.load(source)
    payload: dict[str, np.ndarray] = {}
    source_len = None
    for key in data.files:
        value = data[key]
        if isinstance(value, np.ndarray) and value.ndim >= 1:
            source_len = value.shape[0] if source_len is None else source_len
            if value.shape[0] and target_frames != value.shape[0]:
                repeats = int(np.ceil(target_frames / value.shape[0]))
                tiled = np.concatenate([value] * repeats, axis=0)[:target_frames]
                payload[key] = tiled
            else:
                payload[key] = value
        else:
            payload[key] = value
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dest, **payload)
    return dest


def _copy_metadata(source: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source.is_file():
        dest.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        dest.write_text(json.dumps({}), encoding="utf-8")
    return dest


__all__ = ["match_body_template_duration"]
