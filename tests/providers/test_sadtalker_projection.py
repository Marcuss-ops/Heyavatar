"""Unit tests for the 3DMM(50) → LivePortrait(21, 3) projection.

Pure-Python / numpy — no SadTalker import required. Covers:

* :func:`project_3dmm_to_keypoint_delta` shape contract + identity
  preservation (first 50 dims map through, padding dims stay 0).
* :func:`mouth_aperture_from_jaw` clipping into ``[0, 1]``.
* :func:`sadtalker_coefs_to_driving_flat` round-trip produces a
  flat-packed driving tensor with the canonical ``(T, 63)`` layout.
* Rejecting inputs with wrong last-dim raises :class:`ValueError``.

SadTalker inference end-to-end is exercised separately in
:file:`tests/providers/test_audio_bridge.py` via a stub SadTalker in
``sys.modules``.
"""

from __future__ import annotations

import numpy as np
import pytest

from providers.liveportrait.audio_bridge.projection import (
    mouth_aperture_from_jaw,
    project_3dmm_to_keypoint_delta,
    sadtalker_coefs_to_driving_flat,
)
from providers.liveportrait.audio_bridge.types import (
    DMM_EXPRESSION_DIM,
    EXPRESSION_DIM,
    N_KEYPOINTS,
)


# ----------------------------------------------------------------------
# project_3dmm_to_keypoint_delta
# ----------------------------------------------------------------------


def test_project_3dmm_single_frame_preserves_first_dmm_dims():
    """A (50,) coef vector maps to (63,) with the first 50 = input."""
    rng = np.random.default_rng(42)
    dmm = rng.standard_normal(DMM_EXPRESSION_DIM).astype(np.float32)
    delta = project_3dmm_to_keypoint_delta(dmm)
    assert delta.shape == (N_KEYPOINTS * EXPRESSION_DIM,)
    # Identity placeholder: first 50 dims equal input.
    np.testing.assert_allclose(delta[:DMM_EXPRESSION_DIM], dmm, atol=1e-6)
    # Trailing 13 dims are zero-padded.
    np.testing.assert_allclose(delta[DMM_EXPRESSION_DIM:], 0.0, atol=1e-6)


def test_project_3dmm_batch_frame_shape():
    """A (T, 50) batch maps to (T, 63)."""
    T = 25
    rng = np.random.default_rng(123)
    dmm = rng.standard_normal((T, DMM_EXPRESSION_DIM)).astype(np.float32)
    delta = project_3dmm_to_keypoint_delta(dmm)
    assert delta.shape == (T, N_KEYPOINTS * EXPRESSION_DIM)


def test_project_3dmm_wrong_last_dim_raises():
    bad = np.zeros(49, dtype=np.float32)
    with pytest.raises(ValueError, match="last-dim 50"):
        project_3dmm_to_keypoint_delta(bad)


# ----------------------------------------------------------------------
# mouth_aperture_from_jaw
# ----------------------------------------------------------------------


def test_mouth_aperture_zero_jaw_is_zero():
    jaw = np.zeros(3, dtype=np.float32)  # closed mouth
    ap = mouth_aperture_from_jaw(jaw)
    assert float(ap) == 0.0  # ||jaw||=0 → clip(0.0, 0, 1) → 0.0


def test_mouth_aperture_neutral_jaw_is_half():
    jaw = np.asarray([0.6, 0.0, 0.0], dtype=np.float32)  # ||jaw||=0.6
    ap = mouth_aperture_from_jaw(jaw)
    assert ap == pytest.approx(0.75, abs=1e-5)


def test_mouth_aperture_open_jaw_saturates_to_one():
    jaw = np.asarray([1.2, 0.0, 0.0], dtype=np.float32)  # ||jaw||=1.2
    ap = mouth_aperture_from_jaw(jaw)
    assert ap == 1.0


def test_mouth_aperture_batch_into_0_to_1():
    T = 30
    rng = np.random.default_rng(7)
    jaw = rng.uniform(-2.0, 2.0, size=(T, 3)).astype(np.float32)
    ap = mouth_aperture_from_jaw(jaw)
    assert ap.shape == (T,)
    assert float(ap.min()) >= 0.0
    assert float(ap.max()) <= 1.0


# ----------------------------------------------------------------------
# sadtalker_coefs_to_driving_flat
# ----------------------------------------------------------------------


def test_driving_flat_round_trip_shape():
    T = 12
    rng = np.random.default_rng(99)
    exp = rng.standard_normal((T, DMM_EXPRESSION_DIM)).astype(np.float32)
    jaw = rng.uniform(-1.0, 1.0, size=(T, 3)).astype(np.float32)
    delta, aperture = sadtalker_coefs_to_driving_flat(exp, jaw)
    assert delta.shape == (T, N_KEYPOINTS * EXPRESSION_DIM)
    assert aperture.shape == (T,)
    # Flat reshape gives (T, 21, 3), the upstream ``warp_decode`` shape.
    reshaped = delta.reshape(T, N_KEYPOINTS, EXPRESSION_DIM)
    assert reshaped.shape == (T, 21, 3)


def test_driving_flat_constant_jaw_yields_open_mouth():
    """All-jaw ≈ 1.2 → aperture == 1.0 across all frames."""
    T = 5
    exp = np.zeros((T, DMM_EXPRESSION_DIM), dtype=np.float32)
    jaw = np.tile(np.asarray([1.2, 0.0, 0.0], dtype=np.float32), (T, 1))
    _, aperture = sadtalker_coefs_to_driving_flat(exp, jaw)
    np.testing.assert_allclose(aperture, np.ones(T), atol=1e-6)
