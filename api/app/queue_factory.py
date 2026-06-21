"""Queue backend factory used by :func:`api.app.state.lifespan`.

Selects the job-queue implementation from ``settings.queue_backend``:

* ``"redis"`` → :class:`RedisJobQueue` (requires ``settings.redis_url``).
* ``"memory"`` → :class:`InMemoryJobQueue` (single-process fallback).
* anything else → :class:`NullJobQueue` (drop-everything no-op, used
  in smoke tests).
"""

from __future__ import annotations

from src.core.config import Settings
from src.scheduler.queue.memory import InMemoryJobQueue
from src.scheduler.queue.null import NullJobQueue
from src.scheduler.queue.redis import RedisJobQueue


def _build_queue(settings: Settings):
    """Build the queue instance matching ``settings.queue_backend``."""
    if settings.queue_backend == "redis":
        if not settings.redis_url:
            raise RuntimeError("REDIS_URL must be set when queue_backend='redis'.")
        return RedisJobQueue(url=settings.redis_url)
    if settings.queue_backend == "memory":
        return InMemoryJobQueue()
    return NullJobQueue()
