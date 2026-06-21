"""Shared helpers for real-GPU smoke tests.

* :data:`requires_cuda` — pytest marker that skips when CUDA / PyTorch
  is not available. Each scenario file applies this marker so an
  unprovisioned workstation is silently skipped rather than failing.
* :func:`_setup_live_portrait_path` — adds the cloned LivePortrait
  repo to ``sys.path`` and ``PYTHONPATH`` so the upstream package's
  relative ``from .config …`` imports resolve. Idempotent.
* :func:`_test_image` / :func:`_test_audio` — minimal but valid PNG /
  WAV fixtures used by the compile and render flows.
  ``_test_audio`` writes a 1.0s WAV composed of 0.5s silence
  followed by 0.5s of an active 1 kHz tone so the SSD-based mouth
  sync assertion in :file:`tests/smoke/test_real_gpu/test_pipeline.py`
  can compare per-frame pixel SSD between the silence and the
  active-speech windows.
"""

from __future__ import annotations

import math
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from src.core.config import get_settings

# ── guard: skip if CUDA is unavailable ───────────────────────────────


# Sentinel file inside the cloned LivePortrait upstream repo. When this
# is absent we cannot exercise the real-mode pipeline even if CUDA is
# available — the engine would load into a DEGRADED state and every
# assertion would fail in CI. Skip preemptively instead.
_LIVE_PORTRAIT_UPSTREAM_SENTINEL = (
    Path(__file__).resolve().parents[3]
    / "LivePortrait"
    / "src"
    / "live_portrait_pipeline.py"
)


def _collect_skip_reasons() -> list[str]:
    """Collect every reason the real-GPU scenario should be skipped."""
    reasons: list[str] = []
    try:
        import torch  # type: ignore
        if not torch.cuda.is_available():
            reasons.append("CUDA not available")
    except ImportError:
        reasons.append("PyTorch not installed")
    if not _LIVE_PORTRAIT_UPSTREAM_SENTINEL.is_file():
        reasons.append(
            f"LivePortrait upstream repo not cloned at {_LIVE_PORTRAIT_UPSTREAM_SENTINEL.parent.parent}"
        )
    return reasons


_SKIP_REASONS = _collect_skip_reasons()
requires_cuda = pytest.mark.skipif(
    bool(_SKIP_REASONS), reason="; ".join(_SKIP_REASONS) or "real GPU required"
)


# ── LivePortrait upstream path bootstrap ─────────────────────────────


_LIVE_PORTRAIT_REPO = Path(__file__).resolve().parents[3] / "LivePortrait"


def _setup_live_portrait_path() -> None:
    """Add the LivePortrait repo to ``sys.path`` and ``PYTHONPATH``.

    LivePortrait uses relative imports (``from .config ...``) inside
    ``src/``, so the *repo root* (parent of ``src/``) must be on
    ``sys.path`` so ``import src.live_portrait_pipeline`` resolves.
    """
    if str(_LIVE_PORTRAIT_REPO) not in os.environ.get("PYTHONPATH", ""):
        existing = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = (
            f"{_LIVE_PORTRAIT_REPO}{os.pathsep}{existing}"
            if existing
            else str(_LIVE_PORTRAIT_REPO)
        )
    if str(_LIVE_PORTRAIT_REPO) not in sys.path:
        sys.path.insert(0, str(_LIVE_PORTRAIT_REPO))

    # Also register under the env var the adapter's _import_upstream_live_portrait
    # reads from, so it finds the same checkout.
    os.environ.setdefault("HEYAVATAR_LIVE_PORTRAIT_SRC", str(_LIVE_PORTRAIT_REPO))


# Run once at import time so each test_*.py inherits the path env.
_setup_live_portrait_path()


# ── minimal PNG / WAV fixtures ───────────────────────────────────────


def _test_image(tmp_path: Path) -> Path:
    """Write a real-ish PNG for identity preparation."""
    p = tmp_path / "actor.png"
    from PIL import Image

    # Use a real face image from LivePortrait assets to ensure the model has
    # actual face features to detect and warp (otherwise warping a solid red
    # image yields no pixel changes, causing SSD to be near zero).
    src_jpg = Path(__file__).resolve().parents[3] / "LivePortrait" / "assets" / "examples" / "source" / "s0.jpg"
    if src_jpg.is_file():
        img = Image.open(src_jpg)
        img.save(p)
    else:
        img = Image.new("RGB", (256, 256), color=(255, 0, 0))
        img.save(p)
    return p


def _test_audio(tmp_path: Path) -> Path:
    """Write a 1.0s WAV: 0.5s of silence + 0.5s of an active 1 kHz tone.

    Shape: ``0.0–0.5 s`` zero amplitude, ``0.5–1.0 s`` a 1 kHz sine at
    0.5 amplitude (peak int16 ≈ 16384). 16-bit mono, 16 kHz, total
    16 000 samples.

    Used by ``tests/smoke/test_real_gpu/test_pipeline.py`` to assert
    that the rendered mp4 has per-frame motion in the active-speech
    window but (near-)no motion in the silence window.
    """
    p = tmp_path / "speech.wav"
    sample_rate = 16000
    amplitude = 16384
    silence_samples = sample_rate // 2  # 0.5 s
    tone_samples = sample_rate // 2  # 0.5 s
    samples: list[int] = [0] * silence_samples
    for i in range(tone_samples):
        t = i / sample_rate
        # Amplitude modulate at 5 Hz so the RMS envelope varies dynamically.
        # This drives the DSP audio bridge mouth aperture to open and close,
        # producing non-zero frame-to-frame SSD values.
        mod = 0.5 + 0.5 * math.sin(2.0 * math.pi * 5.0 * t)
        samples.append(int(amplitude * mod * math.sin(2.0 * math.pi * 1000.0 * t)))
    data = b"".join(
        struct.pack("<h", max(-32768, min(32767, s))) for s in samples
    )
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(data),
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        len(data),
    )
    p.write_bytes(header + data)
    return p


# ── ffmpeg / video reading helper ────────────────────────────────────


def _ffmpeg_available() -> bool:
    """Return True iff both ``ffmpeg`` and ``ffprobe`` are on PATH.

    Used by the SSD-based mouth-sync gate test in
    :file:`tests/smoke/test_real_gpu/test_pipeline.py` to skip the
    test when ffmpeg is missing (common in some dev environments) so
    the absence doesn't fail loudly instead of being a clean skip.
    """
    import shutil

    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


requires_ffmpeg = pytest.mark.skipif(
    not _ffmpeg_available(), reason="ffmpeg/ffprobe required for mouth-sync SSD gate"
)


def _read_mp4_frames(mp4_path: Path) -> list[np.ndarray]:
    """Decode ``mp4_path`` into a list of per-frame RGB uint8 arrays.

    Uses ``ffmpeg`` via ``subprocess`` so we don't add an OpenCV or
    PyAV dependency just for the test path. Frames are returned in
    ``(H, W, 3)`` ``uint8`` ``RGB`` ordering to match PIL's.
    """
    import subprocess

    if not mp4_path.is_file():
        raise FileNotFoundError(mp4_path)

    # Pipe raw RGBA → numpy. ``-pix_fmt rgb24`` matches test_pipeline's
    # downstream ROI expectations exactly.
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-i",
        str(mp4_path),
        "-vf",
        "format=rgb24",
        "-f",
        "rawvideo",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    raw = proc.stdout
    if not raw:
        return []
    # We don't know the resolution ahead of time; probe via ffprobe.
    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0",
        str(mp4_path),
    ]
    probe = subprocess.run(probe_cmd, capture_output=True, check=True)
    w_str, h_str = probe.stdout.decode().strip().split(",")
    width, height = int(w_str), int(h_str)
    frame_size = width * height * 3
    frames: list[np.ndarray] = []
    for offset in range(0, len(raw), frame_size):
        chunk = raw[offset : offset + frame_size]
        if len(chunk) != frame_size:
            break
        frames.append(np.frombuffer(chunk, dtype=np.uint8).reshape(height, width, 3))
    return frames


# ── fixtures shared across scenario files ────────────────────────────


@pytest.fixture
def real_mode_env(monkeypatch):
    """Force the settings cache into ``HEYAVATAR_MOCK_ENGINE=0``.

    Used by the engine-load and pipeline scenarios to assert that real
    CUDA paths are exercised (the require_cuda marker skips the test
    on machines without GPU, but settings still defaults to mock).
    """
    monkeypatch.setenv("HEYAVATAR_MOCK_ENGINE", "0")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
