"""Shared fixtures for provider-specific tests.

Tiny, dependency-free, deterministic WAV synthesis so AudioBridge tests
do not network or require libsndfile.
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path
from typing import Iterator

import pytest


def _sine_with_silence(seconds: float, freq: float, sample_rate: int) -> bytes:
    samples: list[int] = []
    n = int(seconds * sample_rate)
    for i in range(n):
        # 100 ms silence at start + 1 s sine + 100 ms silence
        if i < sample_rate * 0.1 or i > sample_rate * 1.1:
            samples.append(0)
        else:
            samples.append(
                int(0.6 * 32767 * math.sin(2 * math.pi * freq * (i / sample_rate)))
            )
    return struct.pack("<" + "h" * len(samples), *samples)


def make_test_wav(path: Path, *, duration_seconds: float = 1.2,
                  freq: float = 220.0, sample_rate: int = 16000) -> Path:
    """Write a deterministic ~1.2 s WAV file for AudioBridge tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _sine_with_silence(duration_seconds, freq, sample_rate)
    with wave.open(str(path), "wb") as wh:
        wh.setnchannels(1)
        wh.setsampwidth(2)
        wh.setframerate(sample_rate)
        wh.writeframes(raw)
    return path


@pytest.fixture
def wav_factory(tmp_path: Path) -> Iterator:
    """Return a callable that writes a deterministic WAV at ``tmp_path``."""
    def factory(name: str = "speech.wav", **kwargs) -> Path:
        return make_test_wav(tmp_path / name, **kwargs)
    return factory
