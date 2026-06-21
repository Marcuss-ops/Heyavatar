"""Domain exceptions for the compositing + quality pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contracts.quality_checker import QCResult


class CompositeError(RuntimeError):
    """Raised when the compositing stage fails unrecoverably."""


class EncodingError(RuntimeError):
    """Raised when the audio/video mux or encoding stage fails."""


class QualityError(RuntimeError):
    """Raised when the QC gate rejects the output video.

    Carries the full :class:`~contracts.quality_checker.QCResult` so the
    caller can log detailed metrics without re-running the checks.
    """

    def __init__(self, status: str, result: "QCResult") -> None:
        self.status = status
        self.result = result
        super().__init__(
            f"Quality gate failed: {status} — "
            f"green={result.debug_green_ratio:.4f}, "
            f"black={result.black_frame_ratio:.4f}, "
            f"duration_delta={result.duration_delta_ms:.1f}ms"
        )


__all__ = ["CompositeError", "EncodingError", "QualityError"]
