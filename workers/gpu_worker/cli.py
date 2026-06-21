"""CLI entry point and queue factory for the GPU worker.

* :func:`build_queue` — chooses :class:`InMemoryJobQueue`,
  :class:`NullJobQueue`, or :class:`RedisJobQueue` based on settings.
* :func:`main` — argparse runtime that bootstraps settings, queue,
  repositories, and starts the :class:`GpuWorker` loop.
"""

from __future__ import annotations

import argparse
import os
import sys

from src.core.config import Settings, get_settings
from src.core.logging import configure_logging, get_logger
from src.domain.enums import EngineId
from src.scheduler.queue.memory import InMemoryJobQueue
from src.scheduler.queue.null import NullJobQueue
from src.scheduler.queue.redis import RedisJobQueue
from src.storage.avatar_packs import AvatarPackRepository
from src.storage.jobs.redis import RedisJobRepository
from workers.gpu_worker.telemetry import _start_metrics_server


def build_queue(settings: Settings):
    """Pick the right queue backend per ``settings.queue_backend``."""
    if settings.queue_backend == "redis":
        if not settings.redis_url:
            raise RuntimeError("REDIS_URL must be set when queue_backend='redis'.")
        return RedisJobQueue(url=settings.redis_url)
    if settings.queue_backend == "memory":
        return InMemoryJobQueue()
    return NullJobQueue()


def main() -> int:  # pragma: no cover - manual integration
    parser = argparse.ArgumentParser(description="Run a GPU worker for the avatar engine.")
    parser.add_argument("--engine", required=False,
                        help="Engine id to bind to (default: settings.worker_engine_id).")
    parser.add_argument("--worker-id", required=False,
                        help="Worker id used for queue reservation.")
    args = parser.parse_args()

    configure_logging()
    settings = get_settings()
    engine_id = EngineId.from_string(args.engine or settings.worker_engine_id)
    from providers import PROVIDERS, get_provider
    if engine_id not in PROVIDERS:
        print(f"Unknown provider for engine {engine_id}", file=sys.stderr)
        return 2
    # Stand up Prometheus exposition first so we have a heartbeat
    # metric even if the engine fails to load.
    _start_metrics_server(settings.worker_metrics_port)
    try:
        from src.observability.distributed.tracing import setup_tracing
        # process_role = worker lets the OTLP exporter label spans
        # per process role on the collector side.
        os.environ.setdefault("HEYAVATAR_PROCESS_ROLE", "worker")
        setup_tracing(get_settings())
    except ImportError:
        pass

    queue = build_queue(settings)

    from contracts.job_queue import QueueHandle
    handle = QueueHandle(
        worker_id=args.worker_id or settings.worker_id,
        engine_id=engine_id.value,
        tier="any",
    )
    repo = AvatarPackRepository(root=settings.pack_dir)
    job_repo = (
        RedisJobRepository(url=settings.redis_url)
        if settings.queue_backend == "redis" and settings.redis_url
        else None
    )
    # Build a shared redis client so the distributed-heartbeat thread
    # reuses the same connection as the job-state repo. Lazy-import of
    # `redis` so dev / mockmode runs without the package still start.
    redis_client = None
    if settings.redis_url:
        try:
            import redis  # type: ignore

            redis_client = redis.Redis.from_url(
                settings.redis_url, decode_responses=True
            )
        except Exception:  # pragma: no cover - dev runs without redis
            redis_client = None

    from workers.gpu_worker.worker import GpuWorker
    worker = GpuWorker(
        engine_id=engine_id,
        settings=settings,
        pack_repo=repo,
        queue=queue,
        handle=handle,
        job_repo=job_repo,
        redis_client=redis_client,
    )
    try:
        worker.run()
    except KeyboardInterrupt:
        worker.stop()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
