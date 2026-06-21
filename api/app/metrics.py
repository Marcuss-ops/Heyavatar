"""Prometheus support for the FastAPI app.

Four concerns live here:

* :func:`_start_queue_depth_gatherer` — a daemon thread that polls
  ``state.queue.depth()`` and writes it to the ``heyavatar_queue_depth``
  gauge every 2 s. Low-frequency because the value is intended as a
  dashboard series, not a hot metric.

* :func:`_strip_path_ids` — regex scrubber that collapses UUIDs and
  mocked ``job-XXXX`` / ``id-XXXX`` path parameters so route labels
  have bounded cardinality for Prometheus rollups.

* :func:`_metrics_middleware_factory` — ASGI middleware that records
  per-request latency via :func:`observe_request`.

* :func:`_mount_metrics` — mounts the ``/metrics`` Prometheus exposition
  endpoint. Returns cleanly when :mod:`prometheus_client` is missing
  so the rest of the API keeps working without it.
"""

from __future__ import annotations

import re
import threading
import time
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request

from src.core.config import get_settings

if TYPE_CHECKING:  # avoid runtime cycle with api.app.state
    from api.app.state import AppState


def _start_queue_depth_gatherer(state: "AppState") -> None:
    """Cheap polling thread that updates ``heyavatar_queue_depth``.

    Polling interval is conservatively long (2 s) because the depth
    gauge is intended to be low-frequency and a slow poll means less
    noise in recorded series. The thread targets the root backend and
    a wildcard tier because the in-process :class:`InMemoryJobQueue`
    does not segregate by tier — Prometheus rollups handle the rest.
    """
    backend = state.settings.queue_backend

    def _loop() -> None:
        while True:
            try:
                depth = int(state.queue.depth())
            except Exception:  # pragma: no cover - defensive
                depth = 0
            try:
                from src.observability.metrics.recorders import (
                    set_queue_depth,
                )

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
                from src.observability.metrics.recorders import observe_request

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
