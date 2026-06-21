"""Chunk configuration policy.

Knobs for the audio → chunks pipeline. Held in a frozen dataclass so
the orchestrator can pass the same policy object across worker
processes without copying.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ChunkConfig:
    """Knobs for the audio → chunks pipeline."""

    chunk_seconds: float = 4.0
    overlap_seconds: float = 0.5
    max_chunks: int = 200  # safety cap based on duration, raised from 8
    chunk_retry_max: int = 3  # retry a single failed chunk up to this many times
    chunk_retry_delay_seconds: float = 0.5  # sleep between retries
