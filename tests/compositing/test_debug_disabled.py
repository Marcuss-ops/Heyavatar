"""Tests verifying that debug overlays are completely isolated from runtime output.

These tests create minimal synthetic video inputs, run the compositor with
``debug=False`` (the production default) and assert that:

1. No file is written to any ``debug/`` directory.
2. The runtime output contains no saturated-green pixels.

A second parametrised test runs with ``debug=True`` and asserts that the
debug preview IS written and that the runtime output is STILL green-free.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from contracts.compositor import CompositeRequest
from src.pipeline import OpenCVFaceCompositor
from src.quality.video_quality import debug_green_ratio


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic video builder
# ─────────────────────────────────────────────────────────────────────────────

def _make_video(path: Path, colour: tuple[int, int, int], n_frames: int = 10) -> None:
    """Write a solid-colour video with n_frames frames at 25 fps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 25.0, (64, 64))
    frame = np.full((64, 64, 3), colour, dtype=np.uint8)
    for _ in range(n_frames):
        out.write(frame)
    out.release()


def _make_mask_video(path: Path, value: int, n_frames: int = 10) -> None:
    """Write a single-channel (greyscale-as-BGR) mask video."""
    _make_video(path, (value, value, value), n_frames)


def _make_transforms(path: Path, n_frames: int = 10, width: int = 64, height: int = 64) -> None:
    """Write a face_transforms.npz with bbox centred in the frame."""
    margin = 10
    bbox = np.array([[margin, margin, width - margin, height - margin]] * n_frames, dtype=np.float32)
    np.savez(str(path), bbox=bbox)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDebugIsolation:

    def test_no_debug_dir_created_when_debug_false(self, tmp_path: Path):
        """With debug=False no 'debug/' directory should be created."""
        _make_video(tmp_path / "body.mp4",       (100, 100, 100))
        _make_video(tmp_path / "face.mp4",        (120, 110, 105))
        _make_mask_video(tmp_path / "face_mask.mp4", 200)
        _make_mask_video(tmp_path / "neck_mask.mp4",  80)
        _make_transforms(tmp_path / "transforms.npz")

        output = tmp_path / "runtime" / "composited.mp4"
        compositor = OpenCVFaceCompositor()
        compositor.composite(CompositeRequest(
            body_video          = tmp_path / "body.mp4",
            generated_face_video= tmp_path / "face.mp4",
            face_mask_video     = tmp_path / "face_mask.mp4",
            neck_mask_video     = tmp_path / "neck_mask.mp4",
            face_transforms     = tmp_path / "transforms.npz",
            output_path         = output,
            debug               = False,
        ))

        debug_dir = tmp_path / "debug"
        assert not debug_dir.exists(), (
            f"debug/ directory was created even though debug=False: {debug_dir}"
        )

    def test_runtime_output_has_no_green_when_debug_false(self, tmp_path: Path):
        """Runtime output must be green-free with debug=False."""
        _make_video(tmp_path / "body.mp4",       (100, 100, 100))
        _make_video(tmp_path / "face.mp4",        (120, 110, 105))
        _make_mask_video(tmp_path / "face_mask.mp4", 200)
        _make_mask_video(tmp_path / "neck_mask.mp4",  80)
        _make_transforms(tmp_path / "transforms.npz")

        output = tmp_path / "runtime" / "composited.mp4"
        compositor = OpenCVFaceCompositor()
        compositor.composite(CompositeRequest(
            body_video          = tmp_path / "body.mp4",
            generated_face_video= tmp_path / "face.mp4",
            face_mask_video     = tmp_path / "face_mask.mp4",
            neck_mask_video     = tmp_path / "neck_mask.mp4",
            face_transforms     = tmp_path / "transforms.npz",
            output_path         = output,
            debug               = False,
        ))

        assert output.is_file(), "Compositor did not produce output file"
        cap = cv2.VideoCapture(str(output))
        max_green = 0.0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            max_green = max(max_green, debug_green_ratio(frame))
        cap.release()

        assert max_green == 0.0, (
            f"Runtime output contains green pixels (ratio={max_green:.4f})"
        )

    def test_runtime_output_has_no_green_when_debug_true(self, tmp_path: Path):
        """Runtime output must be green-free even with debug=True."""
        _make_video(tmp_path / "body.mp4",       (100, 100, 100))
        _make_video(tmp_path / "face.mp4",        (120, 110, 105))
        _make_mask_video(tmp_path / "face_mask.mp4", 200)
        _make_mask_video(tmp_path / "neck_mask.mp4",  80)
        _make_transforms(tmp_path / "transforms.npz")

        output = tmp_path / "runtime" / "composited.mp4"
        compositor = OpenCVFaceCompositor()
        compositor.composite(CompositeRequest(
            body_video          = tmp_path / "body.mp4",
            generated_face_video= tmp_path / "face.mp4",
            face_mask_video     = tmp_path / "face_mask.mp4",
            neck_mask_video     = tmp_path / "neck_mask.mp4",
            face_transforms     = tmp_path / "transforms.npz",
            output_path         = output,
            debug               = True,
        ))

        assert output.is_file(), "Compositor did not produce runtime output file"
        cap = cv2.VideoCapture(str(output))
        max_green = 0.0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            max_green = max(max_green, debug_green_ratio(frame))
        cap.release()

        assert max_green == 0.0, (
            f"Runtime output contains green even with debug=True (ratio={max_green:.4f})"
        )

    def test_debug_preview_written_when_debug_true(self, tmp_path: Path):
        """When debug=True a preview video must appear in debug/."""
        _make_video(tmp_path / "body.mp4",       (100, 100, 100))
        _make_video(tmp_path / "face.mp4",        (120, 110, 105))
        _make_mask_video(tmp_path / "face_mask.mp4", 200)
        _make_mask_video(tmp_path / "neck_mask.mp4",  80)
        _make_transforms(tmp_path / "transforms.npz")

        output = tmp_path / "runtime" / "composited.mp4"
        compositor = OpenCVFaceCompositor()
        compositor.composite(CompositeRequest(
            body_video          = tmp_path / "body.mp4",
            generated_face_video= tmp_path / "face.mp4",
            face_mask_video     = tmp_path / "face_mask.mp4",
            neck_mask_video     = tmp_path / "neck_mask.mp4",
            face_transforms     = tmp_path / "transforms.npz",
            output_path         = output,
            debug               = True,
        ))

        debug_dir = tmp_path / "debug"
        debug_files = list(debug_dir.glob("*.mp4")) if debug_dir.exists() else []
        assert len(debug_files) > 0, (
            f"No debug preview files found in {debug_dir} even though debug=True"
        )
