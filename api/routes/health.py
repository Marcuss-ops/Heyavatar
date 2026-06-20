"""Health endpoint for readiness probes."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/ping")
def ping() -> dict:
    return {"status": "pong"}


@router.get("/healthz")
def healthz(request: Request) -> dict:
    state = getattr(request.app.state, "deps", None)
    if state is None:
        return {"status": "ok", "queue_depth": 0}
    return {
        "status": "ok",
        "queue_depth": state.queue.depth(),
        "backend": state.settings.queue_backend,
        "mock_mode": state.settings.mock_engine,
    }
