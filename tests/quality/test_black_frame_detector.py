"""Unit tests for the mean_luminance() + black-frame logic in VideoQualityChecker.

Covers:
- Black frame → luminance ≈ 0
- White frame → luminance ≈ 255
- Mid-grey    → luminance ≈ 128
- VideoQualityChecker correctly counts black frames in a synthetic video
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.quality.video_quality import mean_luminance


# ─────────────────────────────────────────────────────────────────────────────
# mean_luminance() helper tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMeanLuminance:

    def test_black_frame_is_zero(self):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        assert mean_luminance(frame) == pytest.approx(0.0, abs=0.1)

    def test_white_frame_is_255(self):
        frame = np.full((32, 32, 3), 255, dtype=np.uint8)
        assert mean_luminance(frame) == pytest.approx(255.0, abs=1.0)

    def test_midgrey_is_128(self):
        frame = np.full((32, 32, 3), 128, dtype=np.uint8)
        # BGR(128,128,128) → grey level 128
        assert mean_luminance(frame) == pytest.approx(128.0, abs=2.0)

    def test_pure_red_has_low_luminance(self):
        """BGR (0, 0, 255) → luminance ≈ 76 (standard luma coefficients)."""
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[:, :, 2] = 255   # R channel
        lum = mean_luminance(frame)
        # cv2 greyscale uses weighted sum: L = 0.114*B + 0.587*G + 0.299*R
        expected = 0.299 * 255
        assert lum == pytest.approx(expected, abs=5.0)


# ─────────────────────────────────────────────────────────────────────────────
# Black frame ratio integration check (via VideoQualityChecker)
# ─────────────────────────────────────────────────────────────────────────────

def _make_video_with_blacks(
    path: Path,
    total_frames: int = 20,
    black_frames: int = 10,
    size: int = 16,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 25.0, (size, size))
    for i in range(total_frames):
        colour = 0 if i < black_frames else 128
        frame = np.full((size, size, 3), colour, dtype=np.uint8)
        out.write(frame)
    out.release()


class TestBlackFrameDetection:

    def test_all_black_video_triggers_failure(self, tmp_path: Path):
        """All-black video → FAILED_QC_BLACK_FRAMES."""
        from contracts.quality_checker import QCRequest
        from src.quality.video_quality import VideoQualityChecker

        video = tmp_path / "black.mp4"
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")  # minimal stub

        _make_video_with_blacks(video, total_frames=10, black_frames=10)

        checker = VideoQualityChecker()
        result = checker.check_quality(QCRequest(
            video_path=video,
            audio_path=audio,
            sample_frames=10,
            black_frame_threshold=0.05,
        ))

        assert result.status == "FAILED_QC_BLACK_FRAMES", (
            f"Expected FAILED_QC_BLACK_FRAMES, got {result.status}"
        )
        assert result.black_frame_ratio == pytest.approx(1.0, abs=0.05)

    def test_no_black_frames_passes(self, tmp_path: Path):
        """Video with all grey frames must pass the black-frame check."""
        from contracts.quality_checker import QCRequest
        from src.quality.video_quality import VideoQualityChecker

        video = tmp_path / "grey.mp4"
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

        _make_video_with_blacks(video, total_frames=10, black_frames=0)

        checker = VideoQualityChecker()
        result = checker.check_quality(QCRequest(
            video_path=video,
            audio_path=audio,
            sample_frames=10,
            black_frame_threshold=0.05,
        ))

        # Status may vary on duration check (stub audio); we only care that
        # it does NOT fail specifically on black frames.
        assert result.status != "FAILED_QC_BLACK_FRAMES", (
            f"False positive black-frame failure (status={result.status})"
        )
        assert result.black_frame_ratio == pytest.approx(0.0, abs=0.05)
