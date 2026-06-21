"""Chunk-list manifest writer.

The :class:`EncodingWorker` reads this file to know the in-order
sequence of chunk mp4s (with overlap trimming) and the final output
fps. Lives next to :mod:`use_case` so the orchestrator can call it
without reaching into a separate persistence module.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from src.domain.types import RenderChunkResult, RenderRequest


def _write_chunk_manifest(results: List[RenderChunkResult], request: RenderRequest) -> Path:
    """Write a concat manifest listing chunk paths for the EncodingWorker."""
    dest = Path("./captures") / f"{request.job_id}.manifest.txt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        fh.write(f"# manifest for job {request.job_id}\n")
        fh.write(f"# fps={request.render_spec.fps}\n")
        for r in results:
            fh.write(f"{r.chunk_index}|{r.output_path.resolve().as_posix()}|{r.duration_seconds}\n")
    return dest
