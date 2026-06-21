"""Metrics exposition helpers.

Used by both the FastAPI ``/metrics`` route (in-process registry) and
the standalone worker HTTP server (a private registry per worker).
Tests use :func:`build_private_registry` to get a clean reset between
runs without polluting the global one.
"""

from __future__ import annotations

from typing import Optional, Tuple

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client import REGISTRY as _DEFAULT_REGISTRY


def collect_latest(registry: Optional[CollectorRegistry] = None) -> Tuple[bytes, str]:
    """Return ``(body, content_type)`` for the given registry.

    Pass no argument to use the default Prometheus registry (the
    FastAPI app's process); pass a private registry in tests.
    """
    reg = registry or _DEFAULT_REGISTRY
    return generate_latest(reg), CONTENT_TYPE_LATEST


def build_private_registry() -> CollectorRegistry:
    """Fresh, empty registry so tests don't pollute the global one."""
    reg = CollectorRegistry()
    # Re-register the same metric NAMES but bound to this registry.
    Counter("heyavatar_gpu_seconds_total",
            "test", ["engine_id", "tier"], registry=reg)
    Counter("heyavatar_output_minutes_total",
            "test", ["engine_id", "tier"], registry=reg)
    Gauge("heyavatar_queue_depth", "test", ["backend", "tier"], registry=reg)
    Gauge("heyavatar_inflight_jobs", "test", ["engine_id"], registry=reg)
    Counter("heyavatar_jobs_total", "test", ["state", "tier"], registry=reg)
    Counter("heyavatar_worker_state_transitions_total", "test",
            ["engine_id", "from_state", "to_state"], registry=reg)
    Histogram("heyavatar_request_latency_seconds", "test",
              ["method", "route", "status_class"], registry=reg,
              buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0))
    Histogram("heyavatar_render_chunk_latency_seconds", "test",
              ["engine_id", "degraded"], registry=reg,
              buckets=(0.05, 0.1, 0.25, 0.5, 1.0))
    return reg
