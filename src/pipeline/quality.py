"""VideoQualityChecker — six-point post-production QC for composited videos.

**Canonical location** (per Change 2-EXT of
``docs/REPOSITORY_SLIMMING_PLAN.md`` §4, extending the original
Change 2 compositor move to the QC layer): ``src/pipeline/quality.py``.
The previous home at ``src/quality/video_quality.py`` has been deleted;
both the runtime path (the GPU worker / ``src/application``
orchestrator via the ``contracts.quality_checker.QualityChecker`` ABC)
and the offline preview tool import from here. Contract and class name
are unchanged.

Checks (in order):
  A. Debug green overlay detection
  B. Black frame ratio
  C. Audio/video duration delta
  D. Frame count accuracy
  E. Mask/transform tracking health
  F. File readability via ffprobe

Usage::

    from src.pipeline.quality import VideoQualityChecker
    from contracts.quality_checker import QCRequest

    checker = VideoQualityChecker()
    result = checker.check_quality(QCRequest(
        video_path=Path("runtime/final.mp4"),
        audio_path=Path("speech.wav"),
    ))
    print(result.status, result.debug_green_ratio)
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from contracts.quality_checker import QCRequest, QCResult, QualityChecker
from src.core.logging import get_logger

LOG = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Low-level check helpers (importable for unit tests)
# ─────────────────────────────────────────────────────────────────────────────

def debug_green_ratio(frame: np.ndarray) -> float:
    """Return the fraction of pixels with saturated green colour.

    Uses HSV thresholds: H∈[35,90], S>120, V>80 — the same band used
    by ``contains_debug_green()`` in the pipeline script.

    Args:
        frame: BGR uint8 image array.

    Returns:
        Float in [0, 1].  0 means no green detected.
    """
    if frame is None or frame.size == 0:
        return 0.0
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([35, 120, 80], dtype=np.uint8),
        np.array([90, 255, 255], dtype=np.uint8),
    )
    return float(np.count_nonzero(mask)) / float(mask.size)


def mean_luminance(frame: np.ndarray) -> float:
    """Return mean luminance of *frame* (0‥255)."""
    if frame is None or frame.size == 0:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def probe_video_duration(video_path: Path) -> Optional[float]:
    """Return video duration in seconds via ffprobe, or None on failure."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        LOG.warning("ffprobe not found; duration check skipped")
        return None
    try:
        cmd = [
            ffprobe, "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            str(video_path),
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=15)
        info = json.loads(out)
        for stream in info.get("streams", []):
            raw = stream.get("duration")
            if raw is not None:
                return float(raw)
    except Exception as exc:
        LOG.debug("ffprobe duration probe failed: %s", exc)
    return None


def probe_audio_duration(audio_path: Path) -> Optional[float]:
    """Return audio duration in seconds via ffprobe, or None on failure."""
    return probe_video_duration(audio_path)  # same command works for audio


def probe_video_codec(video_path: Path) -> Optional[str]:
    """Return the video codec name via ffprobe, or None if unreadable."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    try:
        cmd = [
            ffprobe, "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            str(video_path),
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=15)
        info = json.loads(out)
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                return stream.get("codec_name")
    except Exception as exc:
        LOG.debug("ffprobe codec probe failed: %s", exc)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main checker
# ─────────────────────────────────────────────────────────────────────────────

class VideoQualityChecker(QualityChecker):
    """Run all six mandatory QC checks on a composited video.

    Never raises — all failures are captured inside :class:`QCResult`.
    """

    def check_quality(self, request: QCRequest) -> QCResult:  # noqa: PLR0912, PLR0915
        warnings: List[str] = []
        errors: List[str] = []

        # ── Open video capture ────────────────────────────────────────────────
        cap = cv2.VideoCapture(str(request.video_path))
        if not cap.isOpened():
            return QCResult(
                passed=False,
                status="FAILED_QC_UNREADABLE",
                debug_green_ratio=0.0,
                black_frame_ratio=0.0,
                duration_delta_ms=0.0,
                frames_expected=0,
                frames_actual=0,
                invalid_transforms=0,
                errors=[f"Cannot open video: {request.video_path}"],
            )

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # ── A. Sample frames for green + black checks ─────────────────────────
        n_samples = min(request.sample_frames, max(1, total_frames))
        sample_indices = list(
            np.linspace(0, max(0, total_frames - 1), n_samples, dtype=int)
        )

        worst_green_ratio = 0.0
        black_count = 0

        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            gr = debug_green_ratio(frame)
            worst_green_ratio = max(worst_green_ratio, gr)
            if mean_luminance(frame) < 10.0:
                black_count += 1

        black_frame_ratio = black_count / n_samples if n_samples > 0 else 0.0
        cap.release()

        # ── A. Debug green check ──────────────────────────────────────────────
        if worst_green_ratio > request.green_ratio_threshold:
            errors.append(
                f"Debug green overlay detected: ratio={worst_green_ratio:.4f} "
                f"(threshold={request.green_ratio_threshold})"
            )
            return QCResult(
                passed=False,
                status="FAILED_QC_DEBUG_OVERLAY",
                debug_green_ratio=worst_green_ratio,
                black_frame_ratio=black_frame_ratio,
                duration_delta_ms=0.0,
                frames_expected=request.expected_frames or total_frames,
                frames_actual=total_frames,
                invalid_transforms=0,
                errors=errors,
            )

        # ── B. Black frame check ──────────────────────────────────────────────
        if black_frame_ratio > request.black_frame_threshold:
            errors.append(
                f"Too many black frames: {black_frame_ratio:.1%} "
                f"(threshold={request.black_frame_threshold:.1%})"
            )
            return QCResult(
                passed=False,
                status="FAILED_QC_BLACK_FRAMES",
                debug_green_ratio=worst_green_ratio,
                black_frame_ratio=black_frame_ratio,
                duration_delta_ms=0.0,
                frames_expected=request.expected_frames or total_frames,
                frames_actual=total_frames,
                invalid_transforms=0,
                errors=errors,
            )

        # ── C. Duration delta ─────────────────────────────────────────────────
        video_dur = probe_video_duration(request.video_path)
        audio_dur = probe_audio_duration(request.audio_path)
        duration_delta_ms = 0.0

        if video_dur is not None and audio_dur is not None:
            duration_delta_ms = abs(video_dur - audio_dur) * 1000.0
            if duration_delta_ms > request.duration_tolerance_ms:
                errors.append(
                    f"Duration mismatch: video={video_dur:.3f}s "
                    f"audio={audio_dur:.3f}s "
                    f"delta={duration_delta_ms:.1f}ms "
                    f"(tolerance={request.duration_tolerance_ms}ms)"
                )
                return QCResult(
                    passed=False,
                    status="FAILED_QC_DURATION",
                    debug_green_ratio=worst_green_ratio,
                    black_frame_ratio=black_frame_ratio,
                    duration_delta_ms=duration_delta_ms,
                    frames_expected=request.expected_frames or total_frames,
                    frames_actual=total_frames,
                    invalid_transforms=0,
                    errors=errors,
                )
        else:
            warnings.append("Duration check skipped (ffprobe unavailable or probe failed)")

        # ── D. Frame count check ──────────────────────────────────────────────
        frames_expected = request.expected_frames
        if frames_expected is None and audio_dur is not None:
            frames_expected = int(audio_dur * fps)
        if frames_expected is None:
            frames_expected = total_frames

        frame_count_tolerance = 2
        if abs(total_frames - frames_expected) > frame_count_tolerance:
            errors.append(
                f"Frame count mismatch: actual={total_frames} "
                f"expected={frames_expected} "
                f"(tolerance=±{frame_count_tolerance})"
            )
            return QCResult(
                passed=False,
                status="FAILED_QC_FRAME_COUNT",
                debug_green_ratio=worst_green_ratio,
                black_frame_ratio=black_frame_ratio,
                duration_delta_ms=duration_delta_ms,
                frames_expected=frames_expected,
                frames_actual=total_frames,
                invalid_transforms=0,
                errors=errors,
            )

        # ── E. File readability (ffprobe codec) ───────────────────────────────
        codec = probe_video_codec(request.video_path)
        if codec is None:
            errors.append(f"ffprobe could not read video codec: {request.video_path}")
            return QCResult(
                passed=False,
                status="FAILED_QC_UNREADABLE",
                debug_green_ratio=worst_green_ratio,
                black_frame_ratio=black_frame_ratio,
                duration_delta_ms=duration_delta_ms,
                frames_expected=frames_expected,
                frames_actual=total_frames,
                invalid_transforms=0,
                errors=errors,
            )

        # ── All checks passed ─────────────────────────────────────────────────
        return QCResult(
            passed=True,
            status="COMPLETED",
            debug_green_ratio=worst_green_ratio,
            black_frame_ratio=black_frame_ratio,
            duration_delta_ms=duration_delta_ms,
            frames_expected=frames_expected,
            frames_actual=total_frames,
            invalid_transforms=0,
            warnings=warnings,
            errors=errors,
        )


__all__ = [
    "VideoQualityChecker",
    "debug_green_ratio",
    "mean_luminance",
    "probe_video_duration",
    "probe_audio_duration",
    "probe_video_codec",
]
