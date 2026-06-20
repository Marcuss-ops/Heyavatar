"""Job metadata repository — keeps render-job state independent of the queue.

The queue (Redis Streams, in-memory) handles delivery; the repository
handles persistent metadata that survives restarts and is queryable by
id. v1 ships an in-memory implementation; swap it for PostgreSQL later.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from contracts.job_queue import JobState, RenderJob
from src.domain.types import RenderJobId


@dataclass(slots=True)
class InMemoryJobRepository:
    """Thread-safe, in-process job repository."""

    _jobs: Dict[RenderJobId, RenderJob] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def upsert(self, job: RenderJob) -> None:
        with self._lock:
            self._jobs[job.id] = job

    def get(self, job_id: RenderJobId) -> Optional[RenderJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_recent(self, limit: int = 100) -> List[RenderJob]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)[:limit]

    def count_by_state(self) -> Dict[str, int]:
        with self._lock:
            out: Dict[str, int] = {}
            for job in self._jobs.values():
                out[job.state.value] = out.get(job.state.value, 0) + 1
            return out

    def mark(self, job_id: RenderJobId, state: JobState, *, error: Optional[str] = None) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            self._jobs[job_id] = RenderJob(
                id=job.id,
                state=state,
                payload=job.payload,
                created_at=job.created_at,
                updated_at=datetime.now(timezone.utc),
                attempts=job.attempts,
                last_error=error or job.last_error,
                reserved_by=job.reserved_by,
            )
