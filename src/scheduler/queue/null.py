"""Null / no-op job queue.

Disables queueing entirely: every publish is a no-op and
:func:`reserve` always returns ``None``. Useful when you want to run
a worker in --once mode (e.g. manually processing one job) or when
queueing is otherwise undesirable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from contracts.job_queue import (
    JobQueue,
    QueueHandle,
    RenderJob,
)
from src.domain.types import RenderJobId


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
