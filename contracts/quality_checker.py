"""QualityChecker contract — stable interface for post-production validation.

Every quality-checking provider implements :class:`QualityChecker`.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class QCRequest:
    """Inputs for a quality-check run."""

    video_path: Path
    audio_path: Path
    # Expected frame count; if None the checker derives it from fps × duration.
    expected_frames: Optional[int] = None
    # Fraction of saturation-green pixels that triggers FAILED_QC_DEBUG_OVERLAY.
    green_ratio_threshold: float = 0.001
    # Fraction of black frames allowed before failing.
    black_frame_threshold: float = 0.05
    # Maximum allowed absolute drift between video and audio duration (ms).
    duration_tolerance_ms: float = 100.0
    # Number of sampled frames for the green/black checks.
    sample_frames: int = 10


@dataclass(frozen=True)
class QCResult:
    """Detailed outcome of a quality-check run."""

    passed: bool
    # One of: COMPLETED | FAILED_QC_DEBUG_OVERLAY | FAILED_QC_BLACK_FRAMES |
    #         FAILED_QC_DURATION | FAILED_QC_FRAME_COUNT |
    #         FAILED_QC_MASK_TRACKING | FAILED_QC_UNREADABLE
    status: str
    debug_green_ratio: float        # 0‥1 fraction of green pixels in worst sampled frame
    black_frame_ratio: float        # 0‥1 fraction of frames below luminance threshold
    duration_delta_ms: float        # abs(video_duration − audio_duration) in ms
    frames_expected: int
    frames_actual: int
    invalid_transforms: int         # frames where transform had NaN/Inf or was missing
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class QualityChecker(abc.ABC):
    """Contract every quality-checking provider must satisfy."""

    @abc.abstractmethod
    def check_quality(self, request: QCRequest) -> QCResult:
        """Run post-production validation checks on the final MP4.

        Must never raise — failures are captured inside :class:`QCResult`.
        Raises :class:`src.quality.exceptions.QualityError` when
        ``result.passed is False`` if the caller requested hard failure.
        """


__all__ = ["QCRequest", "QCResult", "QualityChecker"]
