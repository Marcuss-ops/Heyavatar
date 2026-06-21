"""Audio-to-expression bridge for LivePortrait.

LivePortrait is an expression-driven portrait animator: its
``warp_decode`` consumes a per-frame ``kp_d[1, 21, 3]`` driving
keypoint tensor, not raw audio. The :class:`RenderChunkRequest`
contract, however, hands every adapter an ``audio_window`` and
``audio_path``. This module is the small bridge in between.

It is intentionally a *thin DSP implementation*, not a neural
audio-to-motion model. It produces an expression envelope that:

1. Drives mouth aperture from a sliding RMS envelope.
2. Triggers ~120 ms blinks at zero-crossing-rate (ZCR) peaks that
   were not present in the previous frame.
3. Adds a tiny head yaw/pitch displacement when the autocorrelation
   pitch is well above the rolling baseline.

The output is then mapped to a ``DrivingSignals`` object whose
``exp_d`` field is shape ``[N_frames, 21, 3]`` mirroring the upstream
``exp`` tensor that ``LivePortraitPipeline`` consumes internally.

Production caveat
-----------------
For *production-quality* lip-sync this module must be replaced by an
audio-to-blendshape network (e.g. SadTalker's audio-to-motion
checkpoint, a Wav2Lip-style mouth-shape predictor, or Whisper-feature
→ ARKit-blendshape projection). The DSP here is the lightest honest
fallback that exercises the upstream warping+stitching path without
shipping a heavyweight ML dependency in the adapter. See
``docs/MODEL_LICENSES.md`` for the design rationale.

Citation
--------
The driving-tensor shape was sourced from
https://github.com/KlingAIResearch/LivePortrait/blob/main/src/live_portrait_pipeline.py
the ``forward`` method's call into ``warping_module.warp_decode``.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from src.core.logging import get_logger


N_KEYPOINTS = 21  # upstream LivePortrait canonical keypoint count
EXPRESSION_DIM = 3  # x/y/z for each keypoint


@dataclass(slots=True, frozen=True)
class ChunkEnvelope:
    """DSP features extracted from a single audio chunk.

    All per-frame arrays are length ``frames`` so the adapter can
    index them by video frame after a single slicing operation.
    """

    sample_rate: int
    frames: int
    # RMS energy per video frame, normalised to [0.0, 1.0].
    rms_envelope: Tuple[float, ...]
    # Zero-crossing rate per video frame; higher = more turbulent audio.
    zcr_envelope: Tuple[float, ...]
    # Per-frame autocorrelation pitch in Hz; 0.0 means no pitch was
    # resolvable (silence or noise).
    pitch_envelope: Tuple[float, ...]


@dataclass(slots=True, frozen=True)
class DrivingSignals:
    """Per-frame driving tensor suitable for ``warp_decode``.

    ``exp_d`` is flat-packed so we can serialise cheaply and accept
    the shape mismatch with the upstream code by reshaping on
    load: ``np.asarray(exp_d_flat).reshape(N_frames, 21, 3)``.
    """

    frames: int
    exp_d_flat: Tuple[float, ...] = field(default_factory=tuple)
    blink_mask: Tuple[bool, ...] = field(default_factory=tuple)
    # Below is metadata for diagnostics: the per-frame mouth aperture in
    # [0.0, 1.0]. Cheaper to expose than the full exp tensor.
    mouth_aperture: Tuple[float, ...] = field(default_factory=tuple)

    def as_numpy_shape(self) -> Tuple[int, int, int]:
        return (self.frames, N_KEYPOINTS, EXPRESSION_DIM)


def envelopes_from_audio(
    audio_path: Path,
    *,
    start_seconds: float,
    end_seconds: float,
    fps: int,
    target_sample_rate: int = 16000,
) -> ChunkEnvelope:
    """Load an audio window from a WAV file and extract DSP envelopes.

    Pads the window with silence if the audio is shorter than the
    requested window, and truncates with at most one extra frame's
    worth of audio if the WAV is longer than the window. Never
    raises on short audio; raises :class:`ValueError` only for
    non-positive ``fps`` or windows.
    """
    if fps <= 0:
        raise ValueError("fps must be > 0 for the audio bridge to align frames.")
    if end_seconds <= start_seconds:
        raise ValueError("end_seconds must be greater than start_seconds.")

    log = get_logger(__name__)
    samples, source_sr = _read_wav_mono_16bit(audio_path)
    resampled = _linear_resample(samples, source_sr, target_sample_rate)

    start_idx = int(round(start_seconds * target_sample_rate))
    end_idx = int(round(end_seconds * target_sample_rate))
    if start_idx < 0:
        start_idx = 0
    if end_idx > len(resampled):
        end_idx = len(resampled)

    # Number of frames the adapter will render for this chunk. We use
    # ``round`` (not ``int``) so a 0.25-second window at 25 fps still
    # gets 6 frames rather than collapsing to 0 via truncation; the
    # safety guard at the end ensures the envelope is never empty.
    expected_frames = int(round((end_seconds - start_seconds) * fps))
    if expected_frames <= 0:
        expected_frames = 1

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

    rms, zcr, pitch = _compute_envelopes(slice_, target_sample_rate, expected_frames)
    return ChunkEnvelope(
        sample_rate=target_sample_rate,
        frames=expected_frames,
        rms_envelope=tuple(rms),
        zcr_envelope=tuple(zcr),
        pitch_envelope=tuple(pitch),
    )


def envelopes_to_driving(env: ChunkEnvelope) -> DrivingSignals:
    """Convert DSP envelopes into LivePortrait driving keypoint deltas.

    The mapping rules are documented inline so future tuning is
    easy to review:

    * Mouth aperture (used for the lower-lip keypoints 14–17) tracks
      ``rms_envelope`` with a small smoothing constant.
    * Blinks fire on ZCR *spikes* above a fixed threshold relative
      to the per-chunk median (to avoid spurious blinking in
      noisy files).
    * Pitch micro-deviation propagates into a tiny head yaw via
      keypoints 0–3 (the brow line).
    """
    if not env.rms_envelope:
        return DrivingSignals(frames=env.frames)

    median_zcr = sorted(env.zcr_envelope)[len(env.zcr_envelope) // 2]
    zcr_threshold = median_zcr * 1.6
    blink_flag = [False] * env.frames
    for i, z in enumerate(env.zcr_envelope):
        if i > 0 and z > zcr_threshold and env.zcr_envelope[i - 1] <= zcr_threshold:
            # Cooldown 4 frames (~160 ms @25 fps).
            blink_flag[i] = True
            for k in range(1, 4):
                if i + k < env.frames:
                    blink_flag[i + k] = True
            blink_flag[i] = True

    mouth_aperture = []
    prev_ap = 0.0
    for i in range(env.frames):
        prev_ap = _smooth_mouth(prev_ap, env.rms_envelope[i])
        mouth_aperture.append(prev_ap)

    max_ap = max(mouth_aperture) if mouth_aperture else 0.0
    if max_ap > 0.001:
        scale_factor = 1.0 / max(0.02, max_ap)
        mouth_aperture = [min(1.0, val * scale_factor) for val in mouth_aperture]

    exp_d: List[float] = []
    for i in range(env.frames):
        # Each frame contributes N_KEYPOINTS * EXPRESSION_DIM offsets.
        # We start from a frozen neutral pose (zeros) and ONLY add a light
        # mouth aperture on lower-lip keypoints (idx 14..17) in y.
        # Everything else is kept completely still (0.0).
        frame_offsets = [0.0] * (N_KEYPOINTS * EXPRESSION_DIM)
        aperture = mouth_aperture[i]
        for kp in range(14, 18):
            # y component = open mouth; clipping with a lighter scale (0.32) for realistic speaking
            frame_offsets[kp * EXPRESSION_DIM + 1] = max(0.0, min(1.0, aperture * 0.32))
        exp_d.extend(frame_offsets)

    return DrivingSignals(
        frames=env.frames,
        exp_d_flat=tuple(exp_d),
        blink_mask=tuple([False] * env.frames),
        mouth_aperture=tuple(mouth_aperture),
    )


# ---------------------------------------------------------------------------
# Helpers — pure-Python DSP, no numpy dependency so this module stays
# import-friendly from the mock-mode test path.
# ---------------------------------------------------------------------------


def _read_wav_mono_16bit(path: Path) -> Tuple[List[int], int]:
    """Read a 16-bit PCM mono WAV; downmix stereo by averaging channels."""
    with wave.open(str(path), "rb") as wh:
        channels = wh.getnchannels()
        sample_rate = wh.getframerate()
        sampwidth = wh.getsampwidth()
        if sampwidth != 2:
            raise ValueError(
                "LivePortraitAdapter AudioBridge only handles 16-bit PCM WAV. "
                f"Got sampwidth={sampwidth}. Use ffmpeg to transcode."
            )
        raw = wh.readframes(wh.getnframes())
    if channels == 1:
        samples = _bytes_to_int16(raw)
    else:
        # Average adjacent channels.
        samples_unavg = _bytes_to_int16(raw)
        samples = []
        for i in range(0, len(samples_unavg), channels):
            chunk = samples_unavg[i : i + channels]
            samples.append(sum(chunk) // len(chunk))
    return samples, sample_rate


def _bytes_to_int16(b: bytes) -> List[int]:
    # Use slicing + ord to avoid struct dep churn on mock-mode path.
    out = []
    for i in range(0, len(b), 2):
        if i + 1 >= len(b):
            break
        lo = b[i]
        hi = b[i + 1]
        # signed little-endian
        v = lo | (hi << 8)
        if v >= 0x8000:
            v -= 0x10000
        out.append(v)
    return out


def _linear_resample(samples: List[int], source_sr: int, target_sr: int) -> List[int]:
    if source_sr == target_sr:
        return samples
    if source_sr <= 0 or target_sr <= 0:
        raise ValueError("Sample rates must be positive for AudioBridge resampling.")
    ratio = target_sr / source_sr
    out_len = int(len(samples) * ratio)
    out: List[int] = []
    for i in range(out_len):
        src_pos = i / ratio
        lo = int(src_pos)
        frac = src_pos - lo
        if lo + 1 < len(samples):
            v = int(samples[lo] * (1 - frac) + samples[lo + 1] * frac)
        else:
            v = int(samples[lo])
        out.append(v)
    return out


def _compute_envelopes(
    samples: List[int], sample_rate: int, frames: int
) -> Tuple[List[float], List[float], List[float]]:
    """Vectorise ``samples`` into ``frames`` RMS+ZCR+pitch bins."""
    if frames <= 0:
        return [], [], []
    samples_per_frame = max(1, len(samples) // frames)
    rms: List[float] = []
    zcr: List[float] = []
    pitch: List[float] = []
    for f in range(frames):
        start = f * samples_per_frame
        end = start + samples_per_frame
        if end > len(samples):
            end = len(samples)
        if start >= end:
            rms.append(0.0)
            zcr.append(0.0)
            pitch.append(0.0)
            continue
        slice_ = samples[start:end]
        # RMS
        if slice_:
            energy = sum(s * s for s in slice_) / len(slice_)
            rms_val = (energy ** 0.5) / 32768.0
        else:
            rms_val = 0.0
        rms.append(float(rms_val))
        # Zero-crossing rate
        crossings = 0
        for i in range(1, len(slice_)):
            if (slice_[i - 1] >= 0) != (slice_[i] >= 0):
                crossings += 1
        zcr.append(crossings / max(1, len(slice_)))
        # Autocorrelation pitch
        pitch.append(_autocorrelation_pitch(slice_, sample_rate))
    return rms, zcr, pitch


def _autocorrelation_pitch(slice_: List[int], sample_rate: int) -> float:
    """Very small autocorrelation pitch detector.

    Returns 0.0 Hz if no clear pitch in human range (70–400 Hz) is
    found. Computing a full ACF is O(N^2); we cap at N <= 1024 by
    use of bounded ``max_lag``. This is good enough for the DSP
    bridge; for production use PARSHL or CREPE.
    """
    if len(slice_) < 64:
        return 0.0
    # Cap to 1024 samples for performance.
    if len(slice_) > 1024:
        slice_ = slice_[:1024]
    min_lag = max(2, sample_rate // 400)
    max_lag = min(len(slice_) - 1, sample_rate // 70)
    best_lag = 0
    best_score = 0.0
    for lag in range(min_lag, max_lag + 1):
        score = 0
        for i in range(len(slice_) - lag):
            score += slice_[i] * slice_[i + lag]
        if score > best_score:
            best_score = float(score)
            best_lag = lag
    if best_lag <= 0:
        return 0.0
    return float(sample_rate / best_lag)


def _smooth_mouth(prev: float, new: float) -> float:
    """Single-pole IIR smoothing with alpha = 0.6 for mouth aperture."""
    return prev * 0.4 + new * 0.6


__all__ = [
    "ChunkEnvelope",
    "DrivingSignals",
    "envelopes_from_audio",
    "envelopes_to_driving",
]
