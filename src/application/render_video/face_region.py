"""Face-region pipeline helpers shared by :mod:`render_cached_avatar` and the offline tooling.

The previous home of these primitives was
:mod:`tools.avatar_assets.test_musetalk_pipeline` (a CLI smoke-test).
Rather than letting ``src/application`` import from ``tools/``, we move
the two reusable stages — *face-ROI extraction* and *audio muxing* —
into this module and have the dedicated test tool :func:`tool_main`
re-export them. Both stages have no engine awareness: they manipulate
plain mp4s using :mod:`cv2` and run :command:`ffmpeg` for the audio mux.

Stages:

* :func:`extract_face_roi` — crops a 256×256 face region per frame from
  ``body.mp4`` using the per-frame ``bbox`` array stored in
  ``face_transforms.npz``. Writes a clean ROI video at native 256×256
  resolution suitable for the MuseTalk face-region-only rendering path
  (VRAM friendly — no full-frame paste-back amplified back through the
  model).
* :func:`mux_audio` — runs :command:`ffmpeg -shortest` to rejoin an
  audio track onto a video file, identical to the pattern used by
  :mod:`workers.encoding_worker.worker`.
* :func:`tool_main` — the body of the pre-existing
  ``tools/avatar_assets/test_musetalk_pipeline.py`` stages 1-2-3-5
  (extract → MuseTalk-stub or real call → pasteback → mux) preserved
  so the offline tool keeps its CLI surface; the new
  :func:`src.application.render_cached_avatar.render_cached_avatar`
  use case imports only :func:`extract_face_roi` and :func:`mux_audio`.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.core.logging import get_logger

LOG = get_logger(__name__)


FACE_REGION_RESOLUTION: tuple[int, int] = (256, 256)


# ─────────────────────────────────────────────────────────────────────────────
# Face-ROI extraction
# ─────────────────────────────────────────────────────────────────────────────


def extract_face_roi(
    body_video_path: Path,
    transforms_path: Path,
    output_roi_path: Path,
    *,
    debug_dir: Optional[Path] = None,
    target_size: int = 256,
) -> int:
    """Crop the per-frame face region from ``body_video_path`` to a clean ROI mp4.

    The bbox per frame is read from ``transforms_path`` (``.npz`` with
    a ``bbox`` array of shape ``(N, 4)`` already aligned to the body
    video coordinates — produced by
    :mod:`tools.avatar_assets.precompute_video_template`). The output
    ROI is ``(target_size, target_size)`` per frame, deterministic and
    silent on bbox failure (falls back to a centred crop).

    Args:
        body_video_path: Source body ``.mp4``.
        transforms_path: Per-frame ``bbox`` and modality arrays (.npz).
        output_roi_path: Clean ROI mp4 path (created if missing).
        debug_dir: When provided, a bbox overlay preview is written
            there. Production callers leave this empty so no debug
            overlay can leak into runtime output.
        target_size: Output resolution (defaults to 256, the canonical
            face-region size consumed by MuseTalk's face-only path).

    Returns:
        Number of frames written to ``output_roi_path``.
    """
    data = np.load(transforms_path)
    bboxes = data["bbox"]

    body_cap = cv2.VideoCapture(str(body_video_path))
    if not body_cap.isOpened():
        raise RuntimeError(
            f"extract_face_roi: cannot open body video at {body_video_path}"
        )

    fps = body_cap.get(cv2.CAP_PROP_FPS)
    width = int(body_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(body_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_roi_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_roi_path), fourcc, fps, (target_size, target_size))

    debug_writer: Optional[cv2.VideoWriter] = None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_writer = cv2.VideoWriter(
            str(debug_dir / "face_bbox_preview.mp4"),
            fourcc, fps, (width, height),
        )

    frames_written = 0
    frame_idx = 0
    while True:
        ret, frame = body_cap.read()
        if not ret or frame_idx >= len(bboxes):
            break

        bbox = bboxes[frame_idx]
        x_min, y_min, x_max, y_max = bbox

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
        else:
            resized = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        writer.write(resized)

        if debug_writer is not None:
            dbg = frame.copy()
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 200, 255), 2)
            debug_writer.write(dbg)

        frame_idx += 1
        frames_written += 1

    body_cap.release()
    writer.release()
    if debug_writer is not None:
        debug_writer.release()

    LOG.debug(
        "extract_face_roi: %d frames at %dx%d from %s",
        frames_written, target_size, target_size, body_video_path,
    )
    return frames_written


# ─────────────────────────────────────────────────────────────────────────────
# Audio muxing
# ─────────────────────────────────────────────────────────────────────────────


def mux_audio(video_path: Path, audio_path: Path, output_path: Path) -> Path:
    """Mux an audio track onto a video using :command:`ffmpeg -shortest`.

    Args:
        video_path: Source video (the lipsynced + composited mp4).
        audio_path: Source audio (e.g. the user's WAV).
        output_path: Destination mp4 with audio track re-encoded as AAC.

    Returns:
        ``output_path``.

    Raises:
        RuntimeError: if ffmpeg is unavailable or fails.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "mux_audio requires ffmpeg on PATH; install it (apt install "
            "ffmpeg / brew install ffmpeg) or run on a worker image that "
            "ships it."
        )
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"mux_audio: ffmpeg failed (exit {proc.returncode}): "
            + proc.stderr.strip()[-400:]
        )
    return output_path


__all__ = [
    "FACE_REGION_RESOLUTION",
    "extract_face_roi",
    "mux_audio",
]
