"""ffmpeg codec resolution for the encoding worker.

Chooses ``h264_nvenc`` when both NVIDIA GPUs are present (via
``nvidia-smi``) and ffmpeg is installed, falling back to ``libx264`` everywhere
else. ``vp9`` maps to ``libvpx-vp9``. Other names are passed through
verbatim (e.g. ``av1`` / ``av1_nvenc``).
"""

from __future__ import annotations

import shutil


def _resolve_codec(name: str) -> str:
    """Map a high-level codec name to a concrete ffmpeg codec."""
    if name == "h264":
        # Use NVIDIA NVENC when available, fall back to libx264 in CI / CPU.
        if shutil.which("nvidia-smi") is not None and shutil.which("ffmpeg"):
            return "h264_nvenc"
        return "libx264"
    if name == "vp9":
        return "libvpx-vp9"
    return name  # passthrough (e.g. 'av1')
