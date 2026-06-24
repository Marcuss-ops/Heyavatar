"""Pure-Python DSP primitives for the audio bridge.

No numpy dependency: this module must be importable in CPU/mock
environments where the rest of the pipeline is exercised without
GPU/weights. ``bridge.py`` composes these primitives into the public
``ChunkEnvelope`` / ``DrivingSignals`` outputs.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import List, Tuple

from src.core.logging import get_logger


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
    """Single-pole IIR smoothing for mouth aperture."""
    return prev * 0.8 + new * 0.2
