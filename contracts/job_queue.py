"""JobQueue abstract interface.

The platform can swap the concrete implementation (Redis Streams, in-memory
single-process for tests, RabbitMQ, SQS) without touching the worker or
the scheduler. Different backends suit different deployment topologies.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from src.domain.types import RenderJobId


class JobState(str, Enum):
    PENDING = "pending"
    RESERVED = "reserved"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_DEGRADED = "completed_degraded"
    FAILED = "failed"
    FAILED_INFERENCE = "failed_inference"
    FAILED_ENCODING = "failed_encoding"
    CANCELLED = "cancelled"


@dataclass(slots=True, frozen=True)
class QueueHandle:
    """Identifier a worker advertises so the router can match capabilities."""

    worker_id: str
    engine_id: str
    tier: str
    vram_total_mb: int = 0


@dataclass(slots=True, frozen=True)
class RenderJob:
    """One entry in the job queue."""

    id: RenderJobId
    state: JobState
    payload: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    attempts: int = 0
    last_error: Optional[str] = None
    reserved_by: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

    def with_state(self, new_state: JobState, *, error: Optional[str] = None, result: Optional[Dict[str, Any]] = None) -> "RenderJob":
        return RenderJob(
            id=self.id,
            state=new_state,
            payload=self.payload,
            created_at=self.created_at,
            updated_at=datetime.now(timezone.utc),
            attempts=self.attempts,
            last_error=error,
            reserved_by=self.reserved_by,
            result=result if result is not None else self.result,
        )


class JobQueue(abc.ABC):
    """Abstract job queue used by GPU workers and the scheduler."""

    name: str

    @abc.abstractmethod
    def publish(self, job: RenderJob) -> None:
        """Add a job to the queue."""

    @abc.abstractmethod
    def reserve(self, handle: QueueHandle) -> Optional[RenderJob]:
        """Atomically pull a job compatible with this handle."""

    @abc.abstractmethod
    def acknowledge(self, job_id: RenderJobId) -> None:
        """Mark a job as completed."""

    @abc.abstractmethod
    def fail(self, job_id: RenderJobId, reason: str) -> None:
        """Mark a job as failed with a reason; the scheduler decides retry policy."""

    @abc.abstractmethod
    def depth(self) -> int:
        """Approximate number of pending jobs."""

    @abc.abstractmethod
    def cancel(self, job_id: RenderJobId) -> None:
        """Set a cancel flag so that workers stop processing this job."""

    def job(self, job_id: RenderJobId) -> Optional[RenderJob]:
        """Optional lookup; defaults to None if the backend doesn't support it."""
        return None


__all__ = ["JobQueue", "RenderJob", "JobState", "QueueHandle"]
