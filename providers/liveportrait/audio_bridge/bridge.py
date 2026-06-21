"""Public audio-to-expression bridge for LivePortrait.

The single canonical entry point is :func:`audio_to_driving`. It
dispatches to the configured backend:

* ``HEYAVATAR_AUDIO_BRIDGE_BACKEND=dsp`` (default for CI / mock mode)
  — pure-Python DSP envelopes (RMS, ZCR, pitch) projected into the
  LivePortrait driving tensor. No ML dependencies. Same motion
  contract as before — reads fine in :file:`tests/providers/test_audio_bridge.py`.
* ``HEYAVATAR_AUDIO_BRIDGE_BACKEND=neural``
  — SadTalker Audio2Motion (3DMM 50+3 coefficients) projected via
  :mod:`providers.liveportrait.audio_bridge.projection` into the
  LivePortrait driving tensor. Production-quality lip-sync. Requires
  ``pip install -e ".[audio-bridge-neural]"`` on the worker image.

Why one function instead of two
------------------------------
The previous split (``envelopes_from_audio`` → ``envelopes_to_driving``)
leaked the DSP-internal :class:`ChunkEnvelope` into the public
boundary. The neural backend has no analogous intermediate. A single
function is the correct public contract; the DSP path keeps its
two-step logic internally and stitches the results before returning.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from src.core.config import Settings, get_settings
from src.core.logging import get_logger

from providers.liveportrait.audio_bridge.dsp import (
    _autocorrelation_pitch,
    _compute_envelopes,
    _linear_resample,
    _read_wav_mono_16bit,
    _smooth_mouth,
)
from providers.liveportrait.audio_bridge.projection import (
    sadtalker_coefs_to_driving_flat,
)
from providers.liveportrait.audio_bridge.types import (
    EXPRESSION_DIM,
    N_KEYPOINTS,
    DrivingSignals,
)


def _validate_window(start_seconds: float, end_seconds: float, fps: int) -> int:
    """Validate the audio window and return ``expected_frames``.

    Raises :class:`ValueError` for non-positive ``fps`` or empty windows;
    returns the canonical frame count so callers can rely on a single
    source of truth for shape maths.
    """
    if fps <= 0:
        raise ValueError("fps must be > 0 for the audio bridge to align frames.")
    if end_seconds <= start_seconds:
        raise ValueError("end_seconds must be greater than start_seconds.")
    expected_frames = int(round((end_seconds - start_seconds) * fps))
    if expected_frames <= 0:
        expected_frames = 1
    return expected_frames


def _audio_to_driving_dsp(
    audio_path: Path,
    *,
    start_seconds: float,
    end_seconds: float,
    fps: int,
    target_sample_rate: int = 16000,
) -> DrivingSignals:
    """DSP-backed implementation of :func:`audio_to_driving`.

    Pure-Python. Mirrors the pre-SadTalker mapping rules so existing
    contract tests stay green and CI without ``mediapipe`` / SadTalker
    still produces a deterministic driving tensor.

    Mapping rules (preserved from the previous bridge.py for
    regression-safety):

    * Mouth aperture (used for the lower-lip keypoints 14-17) tracks
      ``rms_envelope`` with a small smoothing constant.
    * Blinks fire on ZCR *spikes* above a fixed threshold relative
      to the per-chunk median (to avoid spurious blinking in noisy
      files).
    * Pitch micro-deviation propagates into a tiny head yaw via
      keypoints 0-3 (the brow line).

    The mapping is documented inline so future tuning is easy to review.
    """
    expected_frames = _validate_window(start_seconds, end_seconds, fps)
    log = get_logger(__name__)

    samples, source_sr = _read_wav_mono_16bit(audio_path)
    resampled = _linear_resample(samples, source_sr, target_sample_rate)

    start_idx = int(round(start_seconds * target_sample_rate))
    end_idx = int(round(end_seconds * target_sample_rate))
    if start_idx < 0:
        start_idx = 0
    if end_idx > len(resampled):
        end_idx = len(resampled)

    if start_idx >= end_idx:
        log.warning(
            "AudioBridge: empty window for %s [%.3f, %.3f]; padding with silence",
            audio_path,
            start_seconds,
            end_seconds,
        )
        slice_ = [0] * (expected_frames * target_sample_rate // fps)
    else:
        slice_ = resampled[start_idx:end_idx]
        expected_samples = expected_frames * target_sample_rate // fps
        if len(slice_) < expected_samples:
            slice_ = slice_ + [0] * (expected_samples - len(slice_))

    rms, zcr, pitch = _compute_envelopes(slice_, target_sample_rate, expected_frames)
    if not rms:
        return DrivingSignals(frames=expected_frames, backend="dsp")

    median_zcr = sorted(zcr)[len(zcr) // 2]
    zcr_threshold = median_zcr * 1.6
    blink_flag = [False] * expected_frames
    for i, z in enumerate(zcr):
        if i > 0 and z > zcr_threshold and zcr[i - 1] <= zcr_threshold:
            blink_flag[i] = True
            for k in range(1, 4):
                if i + k < expected_frames:
                    blink_flag[i + k] = True
            blink_flag[i] = True

    mouth_aperture: List[float] = []
    prev_ap = 0.0
    for i in range(expected_frames):
        prev_ap = _smooth_mouth(prev_ap, rms[i])
        mouth_aperture.append(prev_ap)

    max_ap = max(mouth_aperture) if mouth_aperture else 0.0
    if max_ap > 0.001:
        scale_factor = 1.0 / max(0.02, max_ap)
        mouth_aperture = [min(1.0, val * scale_factor) for val in mouth_aperture]

    exp_d: List[float] = []
    for i in range(expected_frames):
        frame_offsets = [0.0] * (N_KEYPOINTS * EXPRESSION_DIM)
        ap = mouth_aperture[i]
        for kp in range(14, 18):
            frame_offsets[kp * EXPRESSION_DIM + 1] = max(0.0, min(1.0, ap * 0.32))
        exp_d.extend(frame_offsets)

    return DrivingSignals(
        frames=expected_frames,
        exp_d_flat=tuple(exp_d),
        blink_mask=tuple([False] * expected_frames),
        mouth_aperture=tuple(mouth_aperture),
        backend="dsp",
    )


def _audio_to_driving_neural(
    audio_path: Path,
    *,
    start_seconds: float,
    end_seconds: float,
    fps: int,
    checkpoint_dir: Path | None = None,
) -> DrivingSignals:
    """SadTalker-backed implementation of :func:`audio_to_driving`.

    Wraps :func:`providers.liveportrait.audio_bridge.sadtalker.audio_to_3dmm`
    and folds the 3DMM coefs into LivePortrait's driving tensor via
    :mod:`providers.liveportrait.audio_bridge.projection`.

    Raises :class:`RuntimeError` if SadTalker is unimportable — the
    caller (:mod:`providers.liveportrait.adapter._render`)
    surfaces this as ``EngineState.DEGRADED``. See
    :mod:`providers.liveportrait.audio_bridge.sadtalker` for the
    failure-mode rationale.
    """
    expected_frames = _validate_window(start_seconds, end_seconds, fps)

    from providers.liveportrait.audio_bridge import sadtalker

    # SadTalker's ``audio_to_3dmm`` already resamples the coefs tensor
    # to ``(expected_frames, 53)`` so we don't redo the interpolation
    # here. If the SadTalker branch ever diverges from that
    # contract, an ``AssertionError`` will fire below rather than a
    # silent shape mismatch.
    coefs = sadtalker.audio_to_3dmm(
        audio_path,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        fps=fps,
        checkpoint_dir=checkpoint_dir,
    )
    if coefs.shape[0] != expected_frames:
        # Shape mismatch is a value-level error (the SadTalker
        # module has drifted from our contract), not a transient
        # runtime fault — raise ValueError so caller assertions read
        # naturally.
        raise ValueError(
            f"SadTalker audio_to_3dmm returned {coefs.shape[0]} frames; "
            f"expected {expected_frames}. Resampling is the SadTalker "
            f"module's responsibility; do NOT silently rework it here."
        )

    exp_coefs = coefs[:, :50]
    jaw_coefs = coefs[:, 50:53]
    delta, aperture = sadtalker_coefs_to_driving_flat(exp_coefs, jaw_coefs)
    exp_d_flat = delta.reshape(expected_frames, N_KEYPOINTS * EXPRESSION_DIM)
    return DrivingSignals(
        frames=expected_frames,
        exp_d_flat=tuple(float(v) for v in exp_d_flat.reshape(-1).tolist()),
        blink_mask=tuple([False] * expected_frames),
        mouth_aperture=tuple(float(v) for v in aperture.tolist()),
        backend="neural_sadtalker",
    )


def audio_to_driving(
    audio_path: Path,
    *,
    start_seconds: float,
    end_seconds: float,
    fps: int,
    settings: Settings | None = None,
    checkpoint_dir: Path | None = None,
) -> DrivingSignals:
    """Map an audio window to LivePortrait per-frame driving tensors.

    Returns a single :class:`DrivingSignals` containing the flat-packed
    ``(frames, 21, 3)`` expression delta ``exp_d_flat``, the per-frame
    ``mouth_aperture`` ``[0, 1]`` diagnostic, the ``blink_mask``, and
    the ``backend`` provenance string (``"dsp"`` or ``"neural_sadtalker"``).

    Dispatches to the neural or the DSP backend based on
    :attr:`Settings.audio_bridge_backend`. The check is performed at
    call time (not import time) so a worker can change settings
    mid-process via the settings cache.

    Raises:
        ValueError: ``fps`` non-positive or window empty.
        RuntimeError: backend is ``neural`` and SadTalker failed to
            import. The caller should surface this as ``DEGRADED``,
            not silently fall back.
    """
    settings = settings or get_settings()
    if settings.audio_bridge_backend == "neural":
        return _audio_to_driving_neural(
            audio_path,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            fps=fps,
            checkpoint_dir=checkpoint_dir,
        )
    return _audio_to_driving_dsp(
        audio_path,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        fps=fps,
    )


__all__ = ["audio_to_driving", "DrivingSignals"]
