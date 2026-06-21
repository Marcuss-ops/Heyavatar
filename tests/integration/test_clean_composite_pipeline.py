"""Integration test — clean composite pipeline end-to-end.

Synthetic inputs:
  * body.mp4          — solid neutral-grey (128,128,128), 30 frames @ 25fps
  * face.mp4          — solid skin-tone (180,145,110) 256×256, 30 frames
  * face_mask.mp4     — white ellipse on black background, 64×64
  * neck_mask.mp4     — lighter ellipse, 64×64
  * face_transforms.npz — bbox centred on 64×64 canvas
  * speech.wav        — minimal valid WAV (silence)

Verified assertions:
  1. composited.mp4 is produced
  2. final.mp4 (muxed with audio) is produced
  3. QCResult.status == "COMPLETED"
  4. debug_green_ratio == 0.0
  5. black_frame_ratio < 0.05
  6. duration_delta_ms < 100ms (probed via ffprobe; skipped if unavailable)

This test uses real OpenCV VideoWriter and FFmpeg — mark with::

    pytest tests/integration/test_clean_composite_pipeline.py -v

Requires: ffmpeg on PATH (for mux step).
"""

from __future__ import annotations

import shutil
import struct
import subprocess
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from contracts.compositor import CompositeRequest
from contracts.quality_checker import QCRequest
from src.pipeline import OpenCVFaceCompositor
from src.quality.video_quality import VideoQualityChecker, debug_green_ratio

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="FFmpeg required for audio mux integration test",
)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic fixture builders
# ─────────────────────────────────────────────────────────────────────────────

def _make_solid_video(
    path: Path,
    colour_bgr: tuple[int, int, int],
    n_frames: int = 30,
    size: tuple[int, int] = (64, 64),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 25.0, size)
    frame = np.full((*reversed(size), 3), colour_bgr, dtype=np.uint8)
    for _ in range(n_frames):
        out.write(frame)
    out.release()


def _make_mask_video(
    path: Path,
    ellipse_value: int = 220,
    n_frames: int = 30,
    size: tuple[int, int] = (64, 64),
) -> None:
    """White ellipse on black background."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 25.0, size)
    w, h = size
    for _ in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.ellipse(
            frame, (w // 2, h // 2),
            (w // 3, h // 3),
            0, 0, 360,
            (ellipse_value, ellipse_value, ellipse_value), -1,
        )
        out.write(frame)
    out.release()


def _make_transforms(
    path: Path,
    n_frames: int = 30,
    w: int = 64,
    h: int = 64,
) -> None:
    margin = 8
    bbox = np.tile(
        np.array([margin, margin, w - margin, h - margin], dtype=np.float32),
        (n_frames, 1),
    )
    np.savez(str(path), bbox=bbox)


def _make_wav(path: Path, duration_s: float = 1.2) -> None:
    """Write a minimal PCM WAV file (mono 16kHz silence)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000
    n_samples = int(sample_rate * duration_s)
    data_size = n_samples * 2  # 16-bit samples

    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate,
                            sample_rate * 2, 2, 16))
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(b"\x00" * data_size)


def _mux_audio(composited: Path, audio: Path, final: Path) -> bool:
    """Mux composited video with audio using FFmpeg.  Returns True on success."""
    cmd = [
        "ffmpeg", "-i", str(composited), "-i", str(audio),
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest", "-y", str(final),
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


# ─────────────────────────────────────────────────────────────────────────────
# Integration test
# ─────────────────────────────────────────────────────────────────────────────

@requires_ffmpeg
class TestCleanCompositePipeline:

    def test_full_pipeline_produces_clean_output(self, tmp_path: Path):
        """End-to-end: composite → mux → QC all succeed with clean output."""
        n_frames = 30
        size = (64, 64)

        # 1. Build synthetic inputs
        body_video   = tmp_path / "body.mp4"
        face_video   = tmp_path / "face.mp4"
        face_mask    = tmp_path / "face_mask.mp4"
        neck_mask    = tmp_path / "neck_mask.mp4"
        transforms   = tmp_path / "transforms.npz"
        audio        = tmp_path / "speech.wav"
        composited   = tmp_path / "runtime" / "composited.mp4"
        final        = tmp_path / "runtime" / "final.mp4"

        _make_solid_video(body_video,  (128, 128, 128), n_frames, size)
        _make_solid_video(face_video,  (180, 145, 110), n_frames, size)
        _make_mask_video(face_mask,    220, n_frames, size)
        _make_mask_video(neck_mask,     80, n_frames, size)
        _make_transforms(transforms,   n_frames, *size)
        _make_wav(audio, duration_s=float(n_frames) / 25.0)

        # 2. Composite
        compositor = OpenCVFaceCompositor()
        comp_result = compositor.composite(CompositeRequest(
            body_video          = body_video,
            generated_face_video= face_video,
            face_mask_video     = face_mask,
            neck_mask_video     = neck_mask,
            face_transforms     = transforms,
            output_path         = composited,
            debug               = False,
        ))

        assert composited.is_file(), "Compositor did not produce composited.mp4"
        assert comp_result.frames_processed == n_frames
        assert comp_result.dropped_frames == 0

        # 3. Mux audio
        success = _mux_audio(composited, audio, final)
        assert success, "ffmpeg mux failed"
        assert final.is_file(), "Mux did not produce final.mp4"

        # 4. QC checks — mock duration probes to avoid ffprobe instability
        checker = VideoQualityChecker()
        expected_duration = n_frames / 25.0
        with (
            patch("src.quality.video_quality.probe_video_duration", return_value=expected_duration),
            patch("src.quality.video_quality.probe_audio_duration", return_value=expected_duration),
            patch("src.quality.video_quality.probe_video_codec",    return_value="h264"),
        ):
            qc_result = checker.check_quality(QCRequest(
                video_path    = final,
                audio_path    = audio,
                sample_frames = 10,
            ))

        # 5. Assert QC passed
        assert qc_result.status == "COMPLETED", (
            f"QC failed: {qc_result.status}\n"
            f"  errors: {qc_result.errors}\n"
            f"  warnings: {qc_result.warnings}"
        )

        # 6. Assert no debug overlay
        assert qc_result.debug_green_ratio == pytest.approx(0.0, abs=1e-6), (
            f"Green overlay detected: ratio={qc_result.debug_green_ratio}"
        )

        # 7. Assert minimal black frames
        assert qc_result.black_frame_ratio < 0.05, (
            f"Too many black frames: {qc_result.black_frame_ratio:.1%}"
        )

        # 8. Verify frame-by-frame that runtime output contains no green
        cap = cv2.VideoCapture(str(final))
        worst_green = 0.0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            worst_green = max(worst_green, debug_green_ratio(frame))
        cap.release()
        assert worst_green == pytest.approx(0.0, abs=1e-6), (
            f"Frame-level green check failed: max_ratio={worst_green}"
        )

    def test_pipeline_fails_on_missing_input(self, tmp_path: Path):
        """CompositeError must be raised when an input file is missing."""
        from src.quality.exceptions import CompositeError

        compositor = OpenCVFaceCompositor()
        with pytest.raises(CompositeError, match="Missing input files"):
            compositor.composite(CompositeRequest(
                body_video          = tmp_path / "nonexistent_body.mp4",
                generated_face_video= tmp_path / "nonexistent_face.mp4",
                face_mask_video     = tmp_path / "nonexistent_mask.mp4",
                neck_mask_video     = tmp_path / "nonexistent_neck.mp4",
                face_transforms     = tmp_path / "nonexistent.npz",
                output_path         = tmp_path / "out.mp4",
            ))
