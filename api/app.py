"""FastAPI application factory.

The API owns no GPU; it authenticates, validates, publishes jobs to the
:mod:`JobQueue`, and lets the GPU workers (separate processes) do the
heavy work. This file is the composition root for HTTP routes, Pydantic
schemas, and auth dependencies.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

from fastapi import Depends, FastAPI

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
    yield


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
    return app


app = create_app()
