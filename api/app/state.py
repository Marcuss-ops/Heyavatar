"""Application state and the lifespan context manager.

:class:`AppState` is the dependency bag every route handler resolves
through (``request.app.state.deps``). It owns exactly the interface
contracts the handlers depend on: settings, the queue, pack repo, job
repo, and object store.

:func:`lifespan` is the FastAPI ASGI lifespan hook — it constructs the
:class:`AppState` once at startup, optionally installs the OTLP tracing
exporter, and starts the Prometheus queue-depth gatherer when metrics
are enabled. It then yields; on shutdown it shuts tracing back down.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from fastapi import FastAPI

from api.app.metrics import _start_queue_depth_gatherer
from api.app.queue_factory import _build_queue
from src.core.config import Settings, get_settings
from src.core.logging import configure_logging
from src.scheduler.routing.worker_pool import WorkerPool
from src.storage.avatar_packs import AvatarPackRepository
from src.storage.jobs.memory import InMemoryJobRepository
from src.storage.jobs.redis import RedisJobRepository
from src.storage.object_store import build_object_store


@dataclass(slots=True)
class AppState:
    settings: Settings
    queue: object
    pack_repo: AvatarPackRepository
    job_repo: InMemoryJobRepository | RedisJobRepository
    object_store: object
    # In-process capacity mirror of the cluster's GPU workers.
    # Populated in lifespan by two paths:
    #   (a) in-process :class:`workers.gpu_worker.GpuWorker` instances
    #       that register directly with ``worker_pool.register`` (used
    #       when the API and worker share a process for tests); and
    #   (b) distributed GPU processes that publish heartbeats to
    #       Redis and are absorbed via
    #       :meth:`WorkerPool.sync_from_redis` on a periodic gatherer.
    worker_pool: WorkerPool
    # Shared redis client used by the worker-pool sync gatherer.
    # ``None`` when ``queue_backend != 'redis'`` and no heartbeat
    # gathering is needed.
    redis_client: object | None = None


def _build_redis_client(settings: Settings) -> Optional[object]:
    """Lazily connect to Redis from ``settings.redis_url``.

    Returns ``None`` when the env var is unset or the ``redis``
    package is not installed — in either case the API stays fully
    functional (capacity routing just falls back to in-process
    workers only).
    """
    if not settings.redis_url:
        return None
    try:
        import redis  # type: ignore
    except ImportError:
        return None
    try:
        return redis.Redis.from_url(settings.redis_url, decode_responses=True)
    except Exception:  # pragma: no cover - defensive
        return None


def _start_worker_pool_sync_gatherer(state: AppState) -> None:
    """Daemon thread that mirrors distributed heartbeats into state.worker_pool.

    Calls :meth:`WorkerPool.sync_from_redis` every
    ``settings.api_worker_pool_sync_seconds`` so
    :meth:`TierRouter.pick_available` sees cross-process capacity even
    when the API process and the GPU worker processes live on different
    machines. Skipped silently when no redis client is configured.

    Intervals match the worker-side publish period; a transient blip
    on either side is recovered within the same window because the
    Redis-side TTL (15s by default) outlives two consecutive publish
    windows.
    """
    import threading

    if state.redis_client is None:
        return
    pool = state.worker_pool
    client = state.redis_client
    period = max(0.1, float(state.settings.api_worker_pool_sync_seconds))
    stop = {"v": False}

    def _loop() -> None:
        from src.core.logging import get_logger

        log = get_logger("api.app.state.worker_pool_sync")
        while not stop["v"]:
            try:
                updated = pool.sync_from_redis(client)
                if updated:
                    log.debug(
                        "WorkerPool.sync_from_redis: %d record(s) refreshed",
                        updated,
                    )
            except Exception as exc:  # noqa: BLE001 — must never crash API
                log.debug("WorkerPool.sync_from_redis skipped: %s", exc)
            slept = 0.0
            while not stop["v"] and slept < period:
                import time as _time

                _time.sleep(min(0.5, period - slept))
                slept += 0.5

    thread = threading.Thread(
        target=_loop,
        name="api-worker-pool-sync",
        daemon=True,
    )
    thread.start()
    state._wp_sync_thread = thread  # type: ignore[attr-defined]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    settings = get_settings()
    job_repo = (
        RedisJobRepository(url=settings.redis_url)
        if settings.queue_backend == "redis" and settings.redis_url
        else InMemoryJobRepository()
    )
    redis_client = _build_redis_client(settings)
    state = AppState(
        settings=settings,
        queue=_build_queue(settings),
        pack_repo=AvatarPackRepository(root=settings.pack_dir),
        job_repo=job_repo,
        object_store=build_object_store(settings),
        worker_pool=WorkerPool(),
        redis_client=redis_client,
    )
    app.state.deps = state
    # Tracing is opt-in: only initialise the OTLP exporter when
    # ``OTEL_EXPORTER_OTLP_ENDPOINT`` is configured (settings.otel_endpoint).
    try:
        from src.observability.distributed.tracing import setup_tracing

        setup_tracing(settings)
    except ImportError:
        pass
    # Track queue depth periodically so /metrics remains fresh.
    if settings.api_metrics_enabled:
        _start_queue_depth_gatherer(state)
    # Distributed worker capacity: every ``api_worker_pool_sync_seconds``
    # call ``worker_pool.sync_from_redis(redis_client)`` so a remote
    # worker process publishing ``heyavatar:worker:{id}:health``
    # becomes visible to ``TierRouter.pick_available`` even when the
    # API and the worker live on different machines.
    #
    # **FROZEN per Change 3 / ROADMAP.md §1**. The MVP deployment
    # target is `1 API · 1 Redis · 1 GPU worker · 1 encoder path` so
    # cross-process capacity routing is no longer needed. The gatherer
    # function above is preserved for the existing in-process unit
    # tests; when this looks right again we re-enable with one line
    # beneath this comment.
    yield
    # Signal sync gatherer to stop. Daemon thread will exit on the
    # process exit anyway but a graceful join keeps test output clean.
    thread = getattr(state, "_wp_sync_thread", None)
    if thread is not None:
        try:
            thread.join(timeout=2.0)
        except Exception:  # pragma: no cover - defensive
            pass
    try:
        from src.observability.distributed.tracing import shutdown_tracing

        shutdown_tracing()
    except ImportError:
        pass
