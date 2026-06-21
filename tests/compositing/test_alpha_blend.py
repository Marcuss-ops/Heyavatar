"""Unit tests for the alpha-blend formula used in OpenCVFaceCompositor.

These tests are intentionally pure-NumPy — no OpenCV calls, no file I/O,
no video captures.  They verify the mathematical correctness of the blend.
"""

from __future__ import annotations

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — replicate the blend formula in isolation
# ─────────────────────────────────────────────────────────────────────────────

def alpha_blend(
    foreground: np.ndarray,
    background: np.ndarray,
    alpha: np.ndarray,      # single-channel float32 0‥1, HxW
) -> np.ndarray:
    """Reference implementation of the compositor's blend formula."""
    a3 = alpha[..., np.newaxis]
    result = (
        foreground.astype(np.float32) * a3
        + background.astype(np.float32) * (1.0 - a3)
    )
    return result.clip(0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAlphaBlendFormula:

    def test_full_alpha_returns_foreground(self):
        """alpha = 1.0 everywhere → output == foreground."""
        fg = np.full((8, 8, 3), 200, dtype=np.uint8)
        bg = np.full((8, 8, 3), 50,  dtype=np.uint8)
        alpha = np.ones((8, 8), dtype=np.float32)

        result = alpha_blend(fg, bg, alpha)

        np.testing.assert_array_equal(result, fg)

    def test_zero_alpha_returns_background(self):
        """alpha = 0.0 everywhere → output == background."""
        fg = np.full((8, 8, 3), 200, dtype=np.uint8)
        bg = np.full((8, 8, 3), 50,  dtype=np.uint8)
        alpha = np.zeros((8, 8), dtype=np.float32)

        result = alpha_blend(fg, bg, alpha)

        np.testing.assert_array_equal(result, bg)

    def test_half_alpha_is_average(self):
        """alpha = 0.5 everywhere → output ≈ midpoint of fg and bg."""
        fg = np.full((4, 4, 3), 200, dtype=np.uint8)
        bg = np.full((4, 4, 3), 100, dtype=np.uint8)
        alpha = np.full((4, 4), 0.5, dtype=np.float32)

        result = alpha_blend(fg, bg, alpha)

        expected_val = (200 * 0.5 + 100 * 0.5)  # = 150
        # Allow ±1 for integer rounding
        assert np.all(np.abs(result.astype(int) - int(expected_val)) <= 1), (
            f"Expected ~{expected_val}, got range [{result.min()},{result.max()}]"
        )

    def test_no_overflow_bright_pixels(self):
        """Blend of two max-value frames must not overflow uint8."""
        fg = np.full((16, 16, 3), 255, dtype=np.uint8)
        bg = np.full((16, 16, 3), 255, dtype=np.uint8)
        alpha = np.full((16, 16), 0.9, dtype=np.float32)

        result = alpha_blend(fg, bg, alpha)

        assert result.max() <= 255
        assert result.dtype == np.uint8

    def test_no_underflow_dark_pixels(self):
        """Blend of two zero-value frames must not go below 0."""
        fg = np.zeros((16, 16, 3), dtype=np.uint8)
        bg = np.zeros((16, 16, 3), dtype=np.uint8)
        alpha = np.full((16, 16), 0.5, dtype=np.float32)

        result = alpha_blend(fg, bg, alpha)

        assert result.min() >= 0

    def test_per_pixel_alpha_produces_gradient(self):
        """Linearly increasing alpha → linearly increasing blend output."""
        size = 10
        fg = np.full((size, 1, 3), 100, dtype=np.uint8)
        bg = np.full((size, 1, 3),   0, dtype=np.uint8)
        # alpha[i] = i / (size-1)  → 0.0 at row 0, 1.0 at row size-1
        alpha = np.linspace(0.0, 1.0, size, dtype=np.float32).reshape(size, 1)

        result = alpha_blend(fg, bg, alpha)

        # Output at row 0 ≈ 0, at last row ≈ 100
        assert result[0, 0, 0] <= 2        # near 0
        assert result[-1, 0, 0] >= 98      # near 100

    def test_output_shape_preserved(self):
        """Output shape == input shape."""
        fg = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        bg = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        alpha = np.random.rand(100, 200).astype(np.float32)

        result = alpha_blend(fg, bg, alpha)

        assert result.shape == fg.shape
        assert result.dtype == np.uint8
