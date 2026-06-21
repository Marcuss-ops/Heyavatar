"""Static 3DMM → LivePortrait keypoint-delta projection.

LivePortrait upstream and SadTalker both express face motion in the
same 3D Morphable Model (3DMM) ``exp`` basis — 50 coefficients per
frame. The :func:`_project_3dmm_to_keypoint_delta` function below
takes SadTalker's 3DMM coefficients and emits the per-frame
``(21, 3)`` expression delta that ``warp_decode`` consumes.

There are two projection paths:

1. **Static linear map** (this module). A precomputed
   ``W : R^50 -> R^63`` matrix that we apply to every frame. The
   matrix is documented in :data:`_DMM_TO_LP_STATIC_PROJECTOR` as an
   identity placeholder so unit tests can verify the wiring. On a
   fully provisioned GPU box the real ``W`` is computed by running
   LivePortrait's ``motion_extractor`` on a calibration dataset (see
   :mod:`providers.liveportrait.audio_bridge.calibrate` placeholder
   in a follow-up commit).
2. **Dynamic via LivePortrait motion_extractor** (preferred in
   real-mode jobs). The adapter's ``_real_render_chunk_impl`` already
   holds a reference to ``self._wrapper.motion_extractor`` (the
   ``extract_motion`` path on the LP wrapper). Calling it with the
   SadTalker exp coefficients recovers the exact keypoint deltas LP
   would have produced on a real talking head. We document that
   ordering in the integration tests but do not call
   ``motion_extractor`` per-frame from this module — the canonical
   access point is the adapter layer, where the LP wrapper is
   guaranteed to be loaded.

The static projector stays here as the **fallback** when LP's
``motion_extractor`` cannot be reached (mock mode, broken wrapper,
audio-bridge backend probed from a unit test). Its identity
place-holder preserves the 50-dim coefficients mapped into the first
50 of the 63 target dimensions, padded with zeros. This is a
documented placeholder, not a calibrated projection — the real
matrix ships via calibration.

Why we don't store the calibration matrix in the repo
------------------------------------------------------
The W matrix is derived from calibration data and would lock in a
specific LP weight version. We instead leave the calibration as a
follow-up operational step (``make calibrate-projection`` on the GPU
worker) so the matrix is regenerated when the LP weights are bumped.
The regex ``DMM_TO_LP_STATIC_PROJECTOR``-update is gate-linked to
``LIVE_PORTRAIT_PACK_VERSION`` in :class:`inference_config.PackSchema`.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from providers.liveportrait.audio_bridge.types import (
    DMM_EXPRESSION_DIM,
    EXPRESSION_DIM,
    N_KEYPOINTS,
)


# Identity placeholder: preserves the first 50 of the 63 target dims.
# Real W is produced on the GPU worker by calibration. See the module
# docstring for the follow-up command.
_DMM_TO_LP_STATIC_PROJECTOR: np.ndarray = np.zeros(
    (N_KEYPOINTS * EXPRESSION_DIM, DMM_EXPRESSION_DIM), dtype=np.float32
)
# First 50 of 63 dims map directly through the identity (1.0); the rest
# stay 0 since LP exp-space is intrinsically 50-dim and we're padding.
for _i in range(DMM_EXPRESSION_DIM):
    _DMM_TO_LP_STATIC_PROJECTOR[_i, _i] = 1.0


def project_3dmm_to_keypoint_delta(
    dmm_coefs: np.ndarray,
) -> np.ndarray:
    """Project SadTalker 3DMM expression coefficients to LP keypoint delta.

    Args:
        dmm_coefs: ``(T, 50)`` or ``(50,)`` ``float32`` array.

    Returns:
        ``(T, 63)`` or ``(63,)`` ``float32`` flat-packed keypoint delta
        ready to be reshaped to ``(T, 21, 3)`` for ``warp_decode``.
    """
    single = dmm_coefs.ndim == 1
    if single:
        dmm_coefs = dmm_coefs[np.newaxis, :]
    if dmm_coefs.shape[-1] != DMM_EXPRESSION_DIM:
        raise ValueError(
            f"3DMM coefficient vector must have last-dim 50, "
            f"got {dmm_coefs.shape[-1]}."
        )
    delta = dmm_coefs @ _DMM_TO_LP_STATIC_PROJECTOR.T
    delta = delta.astype(np.float32, copy=False)
    if single:
        return delta[0]
    return delta


def mouth_aperture_from_jaw(
    jaw_coefs: np.ndarray,
) -> np.ndarray:
    """Fold SadTalker jaw / pose into a ``[0, 1]`` mouth aperture proxy.

    SadTalker's jaw vector has 3 components representing the lower-jaw
    rotation in radians. We take the L2-magnitude and squash through a
    fixed sigmoid-like scaling so that:

    * ``||jaw|| = 0``     → ``aperture ≈ 0``   (mouth closed)
    * ``||jaw|| ≈ 0.6``   → ``aperture ≈ 0.5`` (neutral)
    * ``||jaw|| ≈ 1.2``   → ``aperture ≈ 1.0`` (fully open)

    The exact scaling is the same piece of "calibrated placeholder"
    geometry as the static projector: the real audio-to-aperture curve
    ships with the calibration step. For the unit tests this proxy is
    deterministic and in-range.
    """
    single = jaw_coefs.ndim == 1
    if single:
        jaw_coefs = jaw_coefs[np.newaxis, :]
    magnitudes = np.linalg.norm(jaw_coefs.astype(np.float32), axis=-1)
    # 0.6 (neutral) → 0.5, 0.6 - 1.2 (open) → 0.5 - 1.0
    apertures = np.clip((magnitudes - 0.6) / 0.6 + 0.5, 0.0, 1.0)
    if single:
        return apertures[0]
    return apertures


def sadtalker_coefs_to_driving_flat(
    exp_coefs: np.ndarray, jaw_coefs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert SadTalker 3DMM coefs to ``(exp_d_flat, mouth_aperture)``.

    Both arrays returned are ``float32`` numpy:

    * ``exp_d_flat``: shape ``(T, 63)`` — ready for
      ``reshape(exp_d_flat, (T, 21, 3))`` before ``warp_decode``.
    * ``mouth_aperture``: shape ``(T,)`` — equals
      :func:`mouth_aperture_from_jaw`.

    This is the single entry point used by
    :mod:`providers.liveportrait.audio_bridge.bridge`.
    """
    delta = project_3dmm_to_keypoint_delta(exp_coefs)
    aperture = mouth_aperture_from_jaw(jaw_coefs)
    return delta, aperture
