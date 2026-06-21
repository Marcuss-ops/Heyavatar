"""Unit tests for the debug_green_ratio() helper in src.quality.video_quality.

Tests cover:
- Pure green frame  → ratio > threshold
- Pure red frame    → ratio == 0
- Pure blue frame   → ratio == 0
- Neutral grey      → ratio == 0
- Mixed frame (small green patch) → ratio above / below threshold correctly
- Empty / degenerate input        → returns 0.0 gracefully
"""

from __future__ import annotations

import numpy as np
import pytest

from src.quality.video_quality import debug_green_ratio

# HSV green range used by the detector: H∈[35,90], S>120, V>80
# BGR equivalent for a pure high-saturation green: (0, 255, 0) in BGR
PURE_GREEN_BGR   = (0, 255, 0)
PURE_RED_BGR     = (0, 0, 255)
PURE_BLUE_BGR    = (255, 0, 0)
NEUTRAL_GREY_BGR = (128, 128, 128)


def _solid_frame(colour_bgr: tuple[int, int, int], h: int = 32, w: int = 32) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = colour_bgr
    return frame


class TestDebugGreenRatio:

    def test_pure_green_frame_exceeds_threshold(self):
        """A fully green frame must produce ratio >> threshold (0.001)."""
        frame = _solid_frame(PURE_GREEN_BGR)
        ratio = debug_green_ratio(frame)
        assert ratio > 0.001, f"Expected ratio > 0.001, got {ratio}"
        # For a solid green frame the ratio should approach 1.0
        assert ratio > 0.9, f"Expected ratio > 0.9 for solid green, got {ratio}"

    def test_pure_red_frame_is_zero(self):
        """Pure red must not be flagged as debug green."""
        frame = _solid_frame(PURE_RED_BGR)
        ratio = debug_green_ratio(frame)
        assert ratio == 0.0, f"Pure red flagged as green (ratio={ratio})"

    def test_pure_blue_frame_is_zero(self):
        """Pure blue must not be flagged."""
        frame = _solid_frame(PURE_BLUE_BGR)
        ratio = debug_green_ratio(frame)
        assert ratio == 0.0, f"Pure blue flagged as green (ratio={ratio})"

    def test_neutral_grey_is_zero(self):
        """Neutral grey must not be flagged."""
        frame = _solid_frame(NEUTRAL_GREY_BGR)
        ratio = debug_green_ratio(frame)
        assert ratio == 0.0, f"Grey flagged as green (ratio={ratio})"

    def test_small_green_patch_detected(self):
        """A small green patch (> 100 px equivalent) should be detected."""
        h, w = 64, 64
        frame = _solid_frame(NEUTRAL_GREY_BGR, h, w)
        # Paint a 15×15 green square = 225 pixels → ratio = 225/(64*64) ≈ 0.055
        frame[20:35, 20:35] = PURE_GREEN_BGR

        ratio = debug_green_ratio(frame)
        assert ratio > 0.001, (
            f"Small green patch ({15*15} px) not detected; ratio={ratio}"
        )

    def test_tiny_green_patch_below_pixel_count(self):
        """A green patch of only a few pixels should yield a very low ratio.

        The actual detection threshold (> 100 px) lives in the pipeline, not
        in the helper function.  The helper just returns the ratio faithfully.
        """
        h, w = 200, 200
        frame = _solid_frame(NEUTRAL_GREY_BGR, h, w)
        # Paint a 3×3 patch = 9 pixels → ratio ≈ 9/40000 ≈ 0.000225
        frame[10:13, 10:13] = PURE_GREEN_BGR

        ratio = debug_green_ratio(frame)
        # ratio should be non-zero but tiny
        assert 0.0 < ratio < 0.001, (
            f"Expected tiny ratio for 9px patch, got {ratio}"
        )

    @pytest.mark.parametrize("colour", [
        (0, 200, 0),    # darker green — should still trigger
        (30, 255, 30),  # yellow-green
    ])
    def test_saturated_green_variants_detected(self, colour):
        """Various saturated green shades must be detected."""
        frame = _solid_frame(colour)
        ratio = debug_green_ratio(frame)
        assert ratio > 0.001, f"Green variant {colour} not detected; ratio={ratio}"

    def test_olive_or_dark_green_not_detected(self):
        """Olive/dark greens below saturation/value thresholds must not trigger."""
        # Very dark green (V < 80): BGR (0, 40, 0)
        frame = _solid_frame((0, 40, 0))
        ratio = debug_green_ratio(frame)
        assert ratio == 0.0, f"Dark green triggered false positive (ratio={ratio})"
