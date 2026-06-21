"""Unit tests for duration validation in VideoQualityChecker.

These tests do not rely on real audio or real ffprobe — instead they mock
``probe_video_duration`` and ``probe_audio_duration`` to inject controlled
duration values, keeping the tests fast and deterministic.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from contracts.quality_checker import QCRequest
from src.pipeline.quality import VideoQualityChecker


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_grey_video(path: Path, n_frames: int = 30, size: int = 16) -> None:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 25.0, (size, size))
    frame = np.full((size, size, 3), 128, dtype=np.uint8)
    for _ in range(n_frames):
        out.write(frame)
    out.release()


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDurationValidation:

    def _run_check(
        self,
        tmp_path: Path,
        video_dur: float,
        audio_dur: float,
        tolerance_ms: float = 100.0,
    ):
        video = tmp_path / "video.mp4"
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
        _make_grey_video(video, n_frames=30)

        checker = VideoQualityChecker()
        request = QCRequest(
            video_path=video,
            audio_path=audio,
            duration_tolerance_ms=tolerance_ms,
            sample_frames=5,
        )

        with (
            patch("src.pipeline.quality.probe_video_duration", return_value=video_dur),
            patch("src.pipeline.quality.probe_audio_duration", return_value=audio_dur),
            patch("src.pipeline.quality.probe_video_codec",    return_value="h264"),
        ):
            return checker.check_quality(request)

    def test_delta_under_50ms_passes(self, tmp_path: Path):
        """50ms delta is well within the 100ms tolerance."""
        result = self._run_check(tmp_path, video_dur=5.000, audio_dur=5.050)
        assert result.status != "FAILED_QC_DURATION", (
            f"50ms delta was incorrectly rejected (status={result.status})"
        )
        assert result.duration_delta_ms == pytest.approx(50.0, abs=1.0)

    def test_delta_exactly_100ms_passes(self, tmp_path: Path):
        """Exactly at tolerance boundary must pass."""
        result = self._run_check(tmp_path, video_dur=5.000, audio_dur=5.100)
        assert result.status != "FAILED_QC_DURATION", (
            f"100ms delta was rejected (status={result.status})"
        )

    def test_delta_200ms_fails(self, tmp_path: Path):
        """200ms delta must fail with FAILED_QC_DURATION."""
        result = self._run_check(tmp_path, video_dur=5.000, audio_dur=5.200)
        assert result.status == "FAILED_QC_DURATION", (
            f"200ms delta was not rejected (status={result.status})"
        )
        assert result.duration_delta_ms == pytest.approx(200.0, abs=1.0)

    def test_delta_1000ms_fails(self, tmp_path: Path):
        """Large 1-second drift must fail."""
        result = self._run_check(tmp_path, video_dur=5.000, audio_dur=6.000)
        assert result.status == "FAILED_QC_DURATION"

    def test_negative_delta_magnitude_checked(self, tmp_path: Path):
        """Negative direction (video longer than audio) also fails."""
        result = self._run_check(tmp_path, video_dur=5.300, audio_dur=5.000)
        assert result.status == "FAILED_QC_DURATION"
        assert result.duration_delta_ms == pytest.approx(300.0, abs=1.0)

    def test_custom_tolerance_respected(self, tmp_path: Path):
        """A tighter tolerance of 50ms must reject 75ms delta."""
        result = self._run_check(tmp_path, video_dur=5.000, audio_dur=5.075, tolerance_ms=50.0)
        assert result.status == "FAILED_QC_DURATION"

    def test_duration_check_skipped_when_probe_unavailable(self, tmp_path: Path):
        """If ffprobe returns None for both durations, skip duration check gracefully."""
        video = tmp_path / "video.mp4"
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
        _make_grey_video(video, n_frames=30)

        checker = VideoQualityChecker()
        request = QCRequest(video_path=video, audio_path=audio, sample_frames=5)

        with (
            patch("src.pipeline.quality.probe_video_duration", return_value=None),
            patch("src.pipeline.quality.probe_audio_duration", return_value=None),
            patch("src.pipeline.quality.probe_video_codec",    return_value="h264"),
        ):
            result = checker.check_quality(request)

        # Must not crash and must not fail on duration
        assert result.status != "FAILED_QC_DURATION"
        assert any("Duration check skipped" in w for w in result.warnings)
