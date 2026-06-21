"""In-memory job queue implementation.

Single-process, threading-safe via an ``RLock``. Useful for tests and
local development; not safe across processes. For production use
:class:`RedisJobQueue` from :mod:`redis`.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, Optional

from contracts.job_queue import (
    JobQueue,
    JobState,
    QueueHandle,
    RenderJob,
)
from src.domain.types import RenderJobId


@dataclass(slots=True)
class InMemoryJobQueue(JobQueue):
    """Single-process JobQueue implementation. Useful for tests and dev.

    Not safe across processes. For production use :class:`RedisJobQueue`.
    """

    name: str = "memory"
    _pending: Deque[RenderJob] = field(default_factory=deque)
    _by_id: Dict[RenderJobId, RenderJob] = field(default_factory=dict)
    _cancelled: set = field(default_factory=set)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def publish(self, job: RenderJob) -> None:
        with self._lock:
            self._pending.append(job)
            self._by_id[job.id] = job

    def reserve(self, handle: QueueHandle) -> Optional[RenderJob]:
        with self._lock:
            while self._pending:
                job = self._pending.popleft()
                if job.id in self._cancelled:
                    continue
                reserved = RenderJob(
                    id=job.id,
                    state=JobState.RESERVED,
                    payload=job.payload,
                    created_at=job.created_at,
                    updated_at=datetime.now(timezone.utc),
                    attempts=job.attempts + 1,
                    reserved_by=handle.worker_id,
                )
                self._by_id[job.id] = reserved
                return reserved
            return None

    def acknowledge(self, job_id: RenderJobId) -> None:
        with self._lock:
            job = self._by_id.get(job_id)
            if job is None:
                return
            self._by_id[job_id] = job.with_state(JobState.COMPLETED)

    def fail(self, job_id: RenderJobId, reason: str) -> None:
        with self._lock:
            job = self._by_id.get(job_id)
            if job is None:
                return
            self._by_id[job_id] = job.with_state(JobState.FAILED, error=reason)

    def depth(self) -> int:
        with self._lock:
            return len(self._pending)

    def cancel(self, job_id: RenderJobId) -> None:
        with self._lock:
            self._cancelled.add(job_id)
            job = self._by_id.get(job_id)
            if job is not None:
                self._by_id[job_id] = job.with_state(JobState.CANCELLED)

    def job(self, job_id: RenderJobId) -> Optional[RenderJob]:
        with self._lock:
            return self._by_id.get(job_id)
