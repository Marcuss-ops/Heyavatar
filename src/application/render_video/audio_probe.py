"""``ffprobe`` wrapper for audio duration discovery.

The orchestrator needs a cheap upper bound on the audio duration so
it can compute ``max_chunks``. We probe via ``ffprobe`` and fall back
to a conservative ``0.0`` if the binary is missing or the file is
unreadable — the orchestrator then caps the chunk count so a missing
ffprobe can't blow the worker into a runaway render loop.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _probe_audio_duration(audio_path: Path) -> float:
    """Return the duration of an audio file in seconds via ffprobe.

    Returns 0.0 if ffprobe is not available or the file cannot be read,
    so callers fall back gracefully.
    """
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return 0.0
    try:
        result = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except (ValueError, subprocess.TimeoutExpired, OSError):
        pass
    return 0.0
