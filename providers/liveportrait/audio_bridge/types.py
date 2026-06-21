"""Passive data structures shared by the audio_bridge submodules.

After the SadTalker integration the public surface collapsed to a
single :class:`DrivingSignals` — the canonical per-frame lip/expression
tensor :func:`providers.liveportrait.audio_bridge.bridge.audio_to_driving`
returns regardless of backend (``dsp`` or ``neural``).

``N_KEYPOINTS`` and ``EXPRESSION_DIM`` are the upstream LivePortrait
canonical: ``21`` keypoints × ``3`` (x/y/z) per frame. The driving
tensor is flat-packed so a no-copy :func:`numpy.asarray` + ``reshape``
gives the upstream ``warp_decode`` the right shape:
``[N_frames, N_KEYPOINTS, EXPRESSION_DIM]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


N_KEYPOINTS = 21  # upstream LivePortrait canonical keypoint count
EXPRESSION_DIM = 3  # x/y/z for each keypoint

# 3DMM (3D Morphable Model) expression-basis dimensionality. Same as
# SadTalker Audio2Motion output and LivePortrait upstream's
# ``motion_extractor`` ``exp`` field.
DMM_EXPRESSION_DIM = 50

# SadTalker Audio2Motion also outputs a 3-dim jaw / pose vector that
# the audio_bridge can fold into LivePortrait's per-frame head-pose
# rotations. Kept as a separate constant so tests can assert on the
# neural backend without pulling in the (heavy) SadTalker module.
DMM_JAW_DIM = 3


@dataclass(slots=True, frozen=True)
class DrivingSignals:
    """Per-frame driving tensor suitable for ``warp_decode``.

    ``exp_d_flat`` is flat-packed so we can serialise cheaply and accept
    the shape mismatch with the upstream code by reshaping on
    load: ``np.asarray(exp_d_flat).reshape(N_frames, 21, 3)``.
    """

    frames: int
    exp_d_flat: Tuple[float, ...] = field(default_factory=tuple)
    blink_mask: Tuple[bool, ...] = field(default_factory=tuple)
    # Below is metadata for diagnostics: the per-frame mouth aperture in
    # [0.0, 1.0]. The neural backend fills this from SadTalker's
    # jaw-coefficient magnitude; the DSP backend fills it from RMS
    # smoothing.
    mouth_aperture: Tuple[float, ...] = field(default_factory=tuple)
    # Provenance signal so mocks, DSP, and the neural path don't get
    # confused downstream. Values: ``"dsp"``, ``"neural_sadtalker"``,
    # ``"neural_sadtalker_unavailable"`` (test-only).
    backend: str = "dsp"

    def as_numpy_shape(self) -> Tuple[int, int, int]:
        return (self.frames, N_KEYPOINTS, EXPRESSION_DIM)


@dataclass(slots=True, frozen=True)
class SadTalkerCoefs:
    """Per-frame intermediate output from SadTalker Audio2Motion.

    Recorded here as a typed intermediate so the projection layer and
    the unit tests can reason about the 3DMM coefficient space
    without importing the (heavy, GPU-only) SadTalker package.

    Attributes:
        exp: ``(T, 50)`` 3DMM expression coefficients.
        jaw: ``(T, 3)`` jaw / pose rotation. Folded into head-pose in
             :func:`_build_driving_keypoints`.
    """

    exp: Tuple[Tuple[float, ...], ...]  # T x 50
    jaw: Tuple[Tuple[float, ...], ...]  # T x 3

    def frames(self) -> int:
        return len(self.exp)
