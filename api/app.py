"""FastAPI application factory.

The API owns no GPU; it authenticates, validates, publishes jobs to the
:mod:`JobQueue`, and lets the GPU workers (separate processes) do the
heavy work. This file is the composition root for HTTP routes, Pydantic
schemas, auth dependencies, and the Prometheus ``/metrics`` exposition
endpoint.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from fastapi import Depends, FastAPI, Request

from src.core.config import Settings, get_settings
from src.core.logging import configure_logging
from src.scheduler.queue import InMemoryJobQueue, NullJobQueue, RedisJobQueue
from src.storage.avatar_packs import AvatarPackRepository
from src.storage.jobs import InMemoryJobRepository
from src.storage.object_store import build_object_store

from api.auth.api_key import require_api_key
from api.routes import avatars, health, jobs


@dataclass(slots=True)
class AppState:
    settings: Settings
    queue: object
    pack_repo: AvatarPackRepository
    job_repo: InMemoryJobRepository
    object_store: object


def _build_queue(settings: Settings):
    if settings.queue_backend == "redis":
        if not settings.redis_url:
            raise RuntimeError("REDIS_URL must be set when queue_backend='redis'.")
        return RedisJobQueue(url=settings.redis_url)
    if settings.queue_backend == "memory":
        return InMemoryJobQueue()
    return NullJobQueue()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    settings = get_settings()
    state = AppState(
        settings=settings,
        queue=_build_queue(settings),
        pack_repo=AvatarPackRepository(root=settings.pack_dir),
        job_repo=InMemoryJobRepository(),
        object_store=build_object_store(settings),
    )
    app.state.deps = state
    # Tracing is opt-in: only initialise the OTLP exporter when
    # ``OTEL_EXPORTER_OTLP_ENDPOINT`` is configured (settings.otel_endpoint).
    try:
        from src.observability.tracing import setup_tracing
        setup_tracing(settings)
    except ImportError:
        pass
    # Track queue depth periodically so /metrics remains fresh.
    if settings.api_metrics_enabled:
        _start_queue_depth_gatherer(state)
    yield
    try:
        from src.observability.tracing import shutdown_tracing
        shutdown_tracing()
    except ImportError:
        pass


def _start_queue_depth_gatherer(state: AppState) -> None:
    """Cheap polling thread that updates ``heyavatar_queue_depth``.

    Polling interval is conservatively long (2 s) because the depth
    gauge is intended to be low-frequency and a slow poll means less
    noise in recorded series. The thread targets the root backend and
    a wildcard tier because the in-process :class:`InMemoryJobQueue`
    does not segregate by tier — Prometheus rollups handle the rest.
    """
    import threading

    backend = state.settings.queue_backend

    def _loop() -> None:
        while True:
            try:
                depth = int(state.queue.depth())
            except Exception:  # pragma: no cover - defensive
                depth = 0
            try:
                from src.observability.metrics import set_queue_depth, record_terminal
                # Single-tier gauge; per-tier pipeline puts tier in
                # the queue payload. Operators can pivot by using the
                # ``heyavatar_jobs_total`` counter for per-tier views.
                set_queue_depth(backend, "express", depth)
                set_queue_depth(backend, "studio", depth)
                set_queue_depth(backend, "premium", depth)
            except ImportError:
                pass
            time.sleep(2.0)

    thread = threading.Thread(target=_loop, name="queue-depth-gatherer", daemon=True)
    thread.start()


def _strip_path_ids(path: str) -> str:
    """Scrub path parameters so a route label has bounded cardinality.

    FastAPI path templates (``/jobs/{job_id}``) become labels after
    this rewrite (``/jobs/{id}``). Anything that looks like a UUID or
    a 26-char mock id is collapsed. We keep the verb-templated shape
    intact.
    """
    import re

    path = re.sub(r"/[0-9a-fA-F-]{20,}", "/{id}", path)
    path = re.sub(r"/job-[0-9a-zA-Z]+", "/job-{id}", path)
    path = re.sub(r"/id-[0-9a-zA-Z]+", "/id-{id}", path)
    return path


def _metrics_middleware_factory(app: FastAPI):
    """Per-request latency middleware that records to Prometheus."""
    if not get_settings().api_metrics_enabled:
        return None

    async def _middleware(request: Request, call_next):
        started = time.monotonic()
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            status_code = 500
            raise
        finally:
            try:
                from src.observability.metrics import observe_request
                route = _strip_path_ids(request.url.path)
                observe_request(
                    method=request.method,
                    route=route,
                    status_code=status_code,
                    started_monotonic=started,
                )
            except ImportError:
                pass

    return _middleware


def _mount_metrics(app: FastAPI) -> None:
    """Mount the Prometheus ``/metrics`` endpoint on the FastAPI app.

    Raises cleanly if ``prometheus_client`` is not installed — the
    rest of the API must keep working.
    """
    try:
        from prometheus_client import make_asgi_app
    except ImportError:
        return
    sub_app = make_asgi_app()
    app.mount("/metrics", sub_app)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Heyavatar Engine",
        version="0.2.0",
        description="Multi-process avatar engine: FastAPI gateway + GPU workers.",
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(jobs.router, dependencies=[Depends(require_api_key)])
    app.include_router(avatars.router, dependencies=[Depends(require_api_key)])
    _mount_metrics(app)
    middleware = _metrics_middleware_factory(app)
    if middleware is not None:
        # type: ignore[arg-type]
        app.middleware("http")(middleware)
    return app


app = create_app()
