"""Compositor contract — stable interface every compositing provider implements.

Application code imports only from this module; no provider is referenced
directly by the rest of the platform.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class CompositeRequest:
    """All inputs needed to composite a lip-synced face onto a body video."""

    body_video: Path
    generated_face_video: Path
    face_mask_video: Path
    neck_mask_video: Path
    face_transforms: Path
    output_path: Path
    debug: bool = False
    # Optional path to a pre-built color profile; reserved for future use.
    color_profile_path: Optional[Path] = None


@dataclass(frozen=True)
class CompositeResult:
    """Summary metrics produced by a completed compositing run."""

    output_path: Path
    frames_processed: int
    dropped_frames: int
    average_mask_area: float        # normalised 0‥1
    debug_overlay_detected: bool    # True only if QC detects green pixels in runtime output


class Compositor(abc.ABC):
    """Contract every face-compositing provider must satisfy."""

    @abc.abstractmethod
    def composite(self, request: CompositeRequest) -> CompositeResult:
        """Composite *request.generated_face_video* onto *request.body_video*.

        The implementation must:
        * Never write debug colours (green, bbox, text) into the runtime output.
        * Write debug previews only when ``request.debug is True``.
        * Return a :class:`CompositeResult` with accurate frame counts.

        Raises :class:`src.quality.exceptions.CompositeError` on failure.
        """


__all__ = ["CompositeRequest", "CompositeResult", "Compositor"]
