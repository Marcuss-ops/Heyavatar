"""FastAPI application factory.

This is the single composition root for the API: it ties together
HTTP routes, Pydantic schemas, auth dependencies, and the Prometheus
``/metrics`` exposition endpoint.

The module-level ``app`` instance is what ``uvicorn api.app:app``
loads in production (see README.md and the Dockerfile). The factory
itself (:func:`create_app`) is re-exported from :mod:`api.app` for the
test-suite convenience.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI

from api.app.metrics import _metrics_middleware_factory, _mount_metrics
from api.app.state import lifespan
from api.auth.api_key import require_api_key
from api.routes import avatars, health, jobs


def create_app() -> FastAPI:
    """Build and wire the FastAPI application."""
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


# Module-level ASGI instance — Uvicorn's ``api.app:app`` resolves here.
app = create_app()
