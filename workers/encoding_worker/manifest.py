"""Chunk-manifest parser used by :class:`EncodingWorker`.

The manifest format written by :func:`RenderVideo` (see
:mod:`src.application.render_video`) is one chunk per line::

    chunk_index|/absolute/path/to/chunk.mp4|duration_seconds

Lines starting with ``#`` and blank lines are ignored. The parsed
entries feed :meth:`EncodingWorker._concat_with_trim` which trims
overlap from non-first chunks before re-stitching via ffmpeg.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple


def _parse_manifest(path: Path) -> List[Tuple[int, Path, float]]:
    """Parse a chunk manifest file.

    Returns a list of ``(chunk_index, chunk_path, duration_seconds)``.
    Failure-tolerant: malformed lines are silently skipped so a
    partially-written manifest from a crashed chunk-render doesn't
    take down the encoder.
    """
    entries: List[Tuple[int, Path, float]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) >= 3:
                entries.append((int(parts[0]), Path(parts[1]), float(parts[2])))
    return entries
