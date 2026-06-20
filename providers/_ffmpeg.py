"""Shared ffmpeg and pack-IO utilities used by all provider adapters.

Extracted from ``providers/liveportrait/adapter.py`` so MuseTalk,
EchoMimic, and future providers don't couple to LivePortrait internals.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# ffmpeg mp4 writers
# ---------------------------------------------------------------------------


FACE_REGION_RESOLUTION: Tuple[int, int] = (256, 256)


def write_dummy_mp4(
    path: Path,
    *,
    duration: float,
    fps: int,
    colour: str = "0x111111",
    resolution: Tuple[int, int] = (512, 512),
) -> None:
    """Write a synthetic ``.mp4`` (used in mock mode + DEGRADED fallback).

    Requires ffmpeg on PATH. Raises descriptive ``RuntimeError`` if
    the encoder is unavailable or fails so the contract test stays
    precise about its environmental needs.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "_write_dummy_mp4 requires ffmpeg to produce a valid mp4 stub. "
            "Install it (brew install ffmpeg / apt install ffmpeg) or "
            "run tests in an environment that bundles it."
        )
    width, height = resolution
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
        "-f", "lavfi",
        "-i", f"color=c={colour}:s={width}x{height}:r={fps}",
        "-t", f"{duration:.3f}",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to produce mock chunk at {path}: {result.stderr.strip()}"
        )


def write_frames_to_mp4(
    frames: list,
    path: Path,
    *,
    fps: int,
    target_resolution: Tuple[int, int],
    hwaccel: bool | None = None,
) -> None:
    """Encode a list of ``[H, W, 3]`` numpy uint8 frames to a single mp4.

    Auto-detects NVIDIA NVENC when ``nvidia-smi`` is on PATH; set
    ``hwaccel=False`` to force libx264, ``hwaccel=True`` to force NVENC.
    """
    if not frames:
        write_dummy_mp4(path, duration=0.5, fps=fps, resolution=target_resolution)
        return
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "_write_frames_to_mp4 requires ffmpeg; install it or run mock."
        )
    # Auto-detect NVENC when not explicitly set.
    if hwaccel is None:
        hwaccel = shutil.which("nvidia-smi") is not None
    codec = "h264_nvenc" if hwaccel else "libx264"
    width, height = target_resolution
    pipe_cmd = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", codec,
        "-pix_fmt", "yuv420p",
        str(path),
    ]
    process = subprocess.Popen(pipe_cmd, stdin=subprocess.PIPE,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE)
    try:
        for f in frames:
            if f.shape[:2] != (height, width):
                try:
                    from PIL import Image
                    im = Image.fromarray(f).resize((width, height))
                    f = np.asarray(im)
                except Exception:
                    f = f[:height, :width]
            process.stdin.write(f.tobytes())
            try:
                process.stdin.flush()
            except (AttributeError, ValueError):
                pass
        process.stdin.close()
        rc = process.wait()
        if rc != 0:
            err = process.stderr.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"ffmpeg failed encoding to {path}: {err}")
    finally:
        if process.poll() is None:
            process.kill()


# ---------------------------------------------------------------------------
# Pack I/O
# ---------------------------------------------------------------------------


def read_pack_entry(pack_path: Path, name: str) -> bytes:
    """Read a single entry from a tar Avatar Pack; raise KeyError if absent."""
    import tarfile

    with tarfile.open(pack_path, mode="r") as tf:
        try:
            member = tf.getmember(name)
        except KeyError as exc:
            raise KeyError(f"Pack entry '{name}' missing in {pack_path}") from exc
        data = tf.extractfile(member)
        if data is None:
            raise KeyError(f"Pack entry '{name}' is not a regular file")
        return data.read()


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def seed_from_path(path: Path) -> int:
    """Deterministic seed from the first 8 bytes of a file."""
    return int.from_bytes(path.read_bytes()[:8].ljust(8, b"\0"), "little")


def json_dump(d: Dict[str, Any]) -> bytes:
    import json
    return json.dumps(d, indent=2).encode("utf-8")


def to_uint8_hwc(tensor: Any) -> np.ndarray:
    """Convert a torch tensor ``[1, 3, H, W]`` to numpy ``[H, W, 3]``."""
    arr = tensor.detach().cpu().numpy()[0]
    arr = arr.transpose(1, 2, 0)
    return (arr * 255).clip(0, 255).astype(np.uint8) if arr.dtype != np.uint8 else arr


# ── backwards-compat aliases (for adapter import migration) ────────

_write_dummy_mp4 = write_dummy_mp4
_write_frames_to_mp4 = write_frames_to_mp4
_read_pack_entry = read_pack_entry
_seed_from_path = seed_from_path
_json_dump = json_dump
_to_uint8_hwc = to_uint8_hwc
