"""Job metadata repository — keeps render-job state independent of the queue.

The queue (Redis Streams, in-memory) handles delivery; the repository
handles persistent metadata that survives restarts and is queryable by
id. v1 ships an in-memory implementation; swap it for PostgreSQL later.

:class:`RedisJobRepository` provides cross-process job state that is
shared between the FastAPI gateway and the GPU worker, backed by the
same Redis instance the queue uses.
"""

from __future__ import annotations

import json
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


@dataclass(slots=True)
class RedisJobRepository:
    """Cross-process job repository backed by a Redis hash.

    Both the FastAPI gateway and the GPU worker read/write to the same
    Redis instance, so ``GET /jobs/{job_id}`` returns up-to-date state
    even after the worker completes a job in another process.

    Job payloads are stored as JSON inside ``heyavatar:job:{job_id}``.
    """

    url: str = "redis://localhost:6379/0"
    _redis: object = field(default=None, init=False, repr=False)
    _client_factory: Optional[object] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError(
                "RedisJobRepository requires the `redis` Python package. "
                "Install with `pip install redis`."
            ) from exc
        if self._client_factory is None:
            self._client_factory = redis.Redis.from_url
        self._redis = self._client_factory(self.url, decode_responses=True)

    def _hash_key(self, job_id: RenderJobId) -> str:
        return f"heyavatar:job:{job_id}"

    def upsert(self, job: RenderJob) -> None:
        self._redis.hset(
            self._hash_key(job.id),
            mapping={
                "job_id": job.id,
                "state": job.state.value,
                "payload": json.dumps(job.payload),
                "created_at": job.created_at.isoformat(),
                "updated_at": job.updated_at.isoformat(),
                "attempts": str(job.attempts),
                "last_error": job.last_error or "",
                "reserved_by": job.reserved_by or "",
            },
        )

    def get(self, job_id: RenderJobId) -> Optional[RenderJob]:
        raw = self._redis.hgetall(self._hash_key(job_id))
        if not raw:
            return None
        return RenderJob(
            id=RenderJobId(raw.get("job_id", job_id)),
            state=JobState(raw.get("state", "pending")),
            payload=json.loads(raw.get("payload", "{}")),
            created_at=datetime.fromisoformat(raw.get("created_at", datetime.now(timezone.utc).isoformat())),
            updated_at=datetime.fromisoformat(raw.get("updated_at", datetime.now(timezone.utc).isoformat())),
            attempts=int(raw.get("attempts", 0)),
            last_error=raw.get("last_error") or None,
            reserved_by=raw.get("reserved_by") or None,
        )

    def mark(self, job_id: RenderJobId, state: JobState, *, error: Optional[str] = None) -> None:
        # NOTE: read-modify-write is not atomic. If two processes call
        # ``mark()`` on the same job simultaneously, one update may be
        # lost. In the current single-worker-per-job model this race is
        # benign; switch to a Redis transaction (MULTI/EXEC) or Lua
        # script if concurrent writers become possible.
        job = self.get(job_id)
        if job is None:
            return
        updated = RenderJob(
            id=job.id,
            state=state,
            payload=job.payload,
            created_at=job.created_at,
            updated_at=datetime.now(timezone.utc),
            attempts=job.attempts,
            last_error=error or job.last_error,
            reserved_by=job.reserved_by,
        )
        self.upsert(updated)


__all__ = ["InMemoryJobRepository", "RedisJobRepository"]
