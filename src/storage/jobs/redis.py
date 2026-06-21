"""Cross-process job-metadata repository backed by Redis.

Both the FastAPI gateway and the GPU worker read/write to the same
Redis instance the queue uses, so ``GET /jobs/{job_id}`` returns
up-to-date state even after the worker completes a job in another
process.

Job payloads are stored as JSON inside ``heyavatar:job:{job_id}``.

Concurrency note
----------------
``mark()`` is a read-modify-write that is NOT atomic. If two processes
call it on the same job simultaneously, one update may be lost. In
the current single-worker-per-job model this race is benign; switch
to a Redis transaction (MULTI/EXEC) or a Lua script when concurrent
writers become possible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

from contracts.job_queue import JobState, RenderJob
from src.domain.types import RenderJobId


@dataclass(slots=True)
class RedisJobRepository:
    """Cross-process job repository backed by a Redis hash."""

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
        mapping = {
            "job_id": job.id,
            "state": job.state.value,
            "payload": json.dumps(job.payload),
            "created_at": job.created_at.isoformat(),
            "updated_at": job.updated_at.isoformat(),
            "attempts": str(job.attempts),
            "last_error": job.last_error or "",
            "reserved_by": job.reserved_by or "",
        }
        if job.result is not None:
            mapping["result"] = json.dumps(job.result)
        self._redis.hset(
            self._hash_key(job.id),
            mapping=mapping,
        )

    def get(self, job_id: RenderJobId) -> Optional[RenderJob]:
        raw = self._redis.hgetall(self._hash_key(job_id))
        if not raw:
            return None
        result_raw = raw.get("result")
        return RenderJob(
            id=RenderJobId(raw.get("job_id", job_id)),
            state=JobState(raw.get("state", "pending")),
            payload=json.loads(raw.get("payload", "{}")),
            created_at=datetime.fromisoformat(
                raw.get(
                    "created_at", datetime.now(timezone.utc).isoformat()
                )
            ),
            updated_at=datetime.fromisoformat(
                raw.get(
                    "updated_at", datetime.now(timezone.utc).isoformat()
                )
            ),
            attempts=int(raw.get("attempts", 0)),
            last_error=raw.get("last_error") or None,
            reserved_by=raw.get("reserved_by") or None,
            result=json.loads(result_raw) if result_raw else None,
        )

    def mark(
        self,
        job_id: RenderJobId,
        state: JobState,
        *,
        error: Optional[str] = None,
        result: Optional[Dict] = None,
    ) -> None:
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
            result=result if result is not None else job.result,
        )
        self.upsert(updated)
