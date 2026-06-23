"""Shared test fixtures for the Heyavatar engine.

Provides a known-valid 1×1 PNG (``PNG_1X1``), a temp workdir
configured for mock mode, and a deterministic WAV writer
(``make_synthetic_wav``) that downstream audio-driven tests reuse so
the sample-rate math doesn't rot in two places.
"""

from __future__ import annotations

import math
import os
import struct
import wave
from pathlib import Path

import pytest


PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a"
    "0000000d49484452"
    "0000000100000001"
    "08060000001f15c4"
    "89"
    "0000000049454e44"
    "ae426082"
)


def make_synthetic_wav(
    path: Path, *, seconds: float = 2.0, freq: float = 220.0,
    sample_rate: int = 16000,
) -> Path:
    """Write a deterministic mono ``freq``-Hz WAV with leading/trailing silence.

    Used by audio-driven tests so ffprobe reports a known, reproducible
    duration that the QC duration gate can compare against the rendered
    body template without wobble.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(seconds * sample_rate)
    samples = []
    for i in range(n):
        silence_start = sample_rate * 0.05
        silence_end = sample_rate * (seconds - 0.1)
        if i < silence_start or i > silence_end:
            samples.append(0)
        else:
            samples.append(
                int(0.6 * 32767 * math.sin(2 * math.pi * freq * (i / sample_rate)))
            )
    raw = struct.pack("<" + "h" * len(samples), *samples)
    with wave.open(str(path), "wb") as wh:
        wh.setnchannels(1)
        wh.setsampwidth(2)
        wh.setframerate(sample_rate)
        wh.writeframes(raw)
    return path


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    """Configure HEYAVATAR_* env vars and return a sandboxed temp dir."""
    os.environ["HEYAVATAR_MOCK_ENGINE"] = "1"
    os.environ["HEYAVATAR_PACK_DIR"] = str(tmp_path / "packs")
    os.environ["HEYAVATAR_CAPTURE_DIR"] = str(tmp_path / "captures")
    os.environ["HEYAVATAR_OBJECT_STORE"] = str(tmp_path / "object_store")
    os.environ["HEYAVATAR_QUEUE_BACKEND"] = "memory"
    os.environ["HEYAVATAR_REGISTRY"] = "registry/models.yaml"
    (tmp_path / "packs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "captures").mkdir(parents=True, exist_ok=True)
    yield tmp_path
