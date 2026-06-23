"""Face-region helpers for the cached-avatar render path."""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from providers._ffmpeg import FACE_REGION_RESOLUTION


def extract_face_roi(
    body_video_path: Path,
    transforms_path: Path,
    output_roi_path: Path,
    debug_dir: Path | None = None,
    *,
    target_size: int = FACE_REGION_RESOLUTION[0],
) -> None:
    data = np.load(transforms_path)
    bboxes = data["bbox"]

    cap = cv2.VideoCapture(str(body_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open body video: {body_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or target_size)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or target_size)

    output_roi_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_roi_path), fourcc, fps, (target_size, target_size))

    debug_writer = None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_writer = cv2.VideoWriter(str(debug_dir / "face_bbox_preview.mp4"), fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(bboxes):
            break
        x_min, y_min, x_max, y_max = bboxes[frame_idx]
        cx = int((x_min + x_max) // 2)
        cy = int((y_min + y_max) // 2)
        size = int(max(x_max - x_min, y_max - y_min) * 1.3)
        x1 = max(0, cx - size // 2)
        y1 = max(0, cy - size // 2)
        x2 = min(width, cx + size // 2)
        y2 = min(height, cy + size // 2)
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            writer.write(resized)
        else:
            writer.write(np.zeros((target_size, target_size, 3), dtype=np.uint8))
        if debug_writer is not None:
            dbg = frame.copy()
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 200, 255), 2)
            debug_writer.write(dbg)
        frame_idx += 1

    cap.release()
    writer.release()
    if debug_writer is not None:
        debug_writer.release()


def mux_audio(video_path: Path, audio_path: Path, output_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        "-y",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
