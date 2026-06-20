"""Concrete job queue implementations.

Three flavours live here:

* :class:`InMemoryJobQueue` — single-process, used for tests and CI.
* :class:`NullJobQueue` — silent queue that drops every job. Useful when you
  want to disable queueing entirely (e.g. running a worker in --once mode).
* :class:`RedisJobQueue` — Redis-Streams-backed, the production default.
  Imported lazily so the project keeps working without Redis installed.

All implementations honour :class:`JobQueue` from :mod:`contracts.job_queue`.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional

from contracts.job_queue import (
    JobQueue,
    JobState,
    QueueHandle,
    RenderJob,
)
from src.domain.types import RenderJobId


# ---------------------------------------------------------------------------
# In-memory
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Null (disable queueing)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class NullJobQueue(JobQueue):
    """Disables queueing: every publish is a no-op, reserve returns None."""

    name: str = "null"

    def publish(self, job: RenderJob) -> None:  # pragma: no cover
        return None

    def reserve(self, handle: QueueHandle) -> Optional[RenderJob]:  # pragma: no cover
        return None

    def acknowledge(self, job_id: RenderJobId) -> None:  # pragma: no cover
        return None

    def fail(self, job_id: RenderJobId, reason: str) -> None:  # pragma: no cover
        return None

    def depth(self) -> int:  # pragma: no cover
        return 0

    def cancel(self, job_id: RenderJobId) -> None:  # pragma: no cover
        return None


# ---------------------------------------------------------------------------
# Redis (lazy import)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RedisJobQueue(JobQueue):
    """Production queue backed by Redis Streams.

    Streams are imported lazily so the package is usable in development
    environments without a Redis server installed.
    """

    name: str = "redis"
    url: str = "redis://localhost:6379/0"
    stream: str = "heyavatar:jobs"
    consumer_group: str = "heyavatar-workers"
    visibility_timeout_seconds: int = 120
    _redis: object = field(default=None, init=False, repr=False)
    _client_factory: Optional[object] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        try:
            import redis  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "RedisJobQueue requires the `redis` Python package. "
                "Install with `pip install redis`."
            ) from exc

        if self._client_factory is None:
            self._client_factory = redis.Redis.from_url
        self._redis = self._client_factory(self.url, decode_responses=True)
        try:
            self._redis.xgroup_create(self.stream, self.consumer_group, mkstream=True)
        except Exception as exc:  # pragma: no cover - group may already exist
            from redis.exceptions import ResponseError
            if not isinstance(exc, ResponseError):
                raise


    def publish(self, job: RenderJob) -> None:
        # Persist the payload in a side-hash so reserve() can reconstruct
        # the full RenderJob. Without this, every reserved job's payload
        # would be empty in production.
        import json
        self._redis.hset(
            f"heyavatar:job:{job.id}",
            mapping={
                "payload": json.dumps(job.payload),
                "state": job.state.value,
            },
        )
        self._redis.xadd(self.stream, {"job_id": job.id, "state": job.state.value})

    def reserve(self, handle: QueueHandle) -> Optional[RenderJob]:
        reply = self._redis.xreadgroup(
            self.consumer_group,
            handle.worker_id,
            streams={self.stream: ">"},
            count=1,
            block=1000,
        )
        if not reply:
            return None
        _, entries = reply[0]
        if not entries:
            return None
        entry_id, fields = entries[0]
        job_id = RenderJobId(fields["job_id"])
        # The full RenderJob is loaded from a side-channel (Redis hash); in the
        # reference implementation we store the JSON-encoded payload alongside
        # the stream.
        payload_raw = self._redis.hget(f"heyavatar:job:{job_id}", "payload")
        import json
        payload = json.loads(payload_raw) if payload_raw else {}
        return RenderJob(
            id=job_id,
            state=JobState.RESERVED,
            payload=payload,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            attempts=1,
            reserved_by=handle.worker_id,
        )

    def acknowledge(self, job_id: RenderJobId) -> None:
        self._redis.hset(f"heyavatar:job:{job_id}", mapping={"state": JobState.COMPLETED.value})

    def fail(self, job_id: RenderJobId, reason: str) -> None:
        self._redis.hset(
            f"heyavatar:job:{job_id}",
            mapping={"state": JobState.FAILED.value, "error": reason},
        )

    def depth(self) -> int:
        return int(self._redis.xlen(self.stream))

    def cancel(self, job_id: RenderJobId) -> None:
        self._redis.hset(f"heyavatar:job:{job_id}", mapping={"state": JobState.CANCELLED.value})


__all__ = ["InMemoryJobQueue", "NullJobQueue", "RedisJobQueue"]
