"""Prometheus metrics facade for the Heyavatar engine.

This module registers a small, *low-cardinality* metric set on the
default Prometheus collector registry. The headline economic metric
— **GPU-seconds per minute of useful output** — is therefore
expressed as TWO counters that operators combine in PromQL with
:func:`rate`. This is the only way to get a correct ratio in
multi-process Prometheus where each worker publishes its own
cumulative totals.

Why Counters, not a Gauge
--------------------------
A precomputed rolling-window ``Gauge`` updated by a Python
background thread is a known anti-pattern:

* It loses data when a worker dies.
* Cross-worker aggregation of precomputed rolling ratios is the
  average-of-averages fallacy.
* It cannot be slice-and-diced in Grafana ("ratio in the last
  30 seconds vs last hour" is impossible without recomputing in
  Prometheus).

Two Counters (``gpu_seconds_total``, ``output_minutes_total``) plus
``rate()`` in the dashboard give us:

* Crash-safe cumulative totals.
* Cross-worker summation that is mathematically correct.
* Arbitrary windowing in Grafana.

Metric cardinality contract
---------------------------
Labels in this module are **strictly** limited to low-cardinality
``engine_id`` (``"musetalk-v1"`` … ``"liveportrait-human-v1"``)
and ``tier`` (``"express"`` / ``"studio"`` / ``"premium"``).
**Never** add ``job_id`` / ``identity_id`` / ``worker_id`` /
``request_path`` to any of these metrics. The historian sidetrack
in ``docs/observability.md`` includes a CIDR-block scraper
configuration that omits high-cardinality labels.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional, Tuple

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client import REGISTRY as _DEFAULT_REGISTRY


# ---------------------------------------------------------------------------
# Constants: low-cardinality label sets.
# ---------------------------------------------------------------------------


ENGINE_IDS = ("musetalk-v1", "liveportrait-human-v1", "echomimic-v1", "mock")
TIERS = ("express", "studio", "premium")
JOB_STATES = ("pending", "reserved", "rendering", "completed", "failed", "cancelled")
WORKER_STATES = ("unloaded", "loading", "idle", "rendering", "degraded", "error")
QUEUE_BACKENDS = ("memory", "null", "redis")
HTTP_METHODS = ("GET", "POST", "DELETE", "PUT", "PATCH")


def _labels_kwargs(engine_id: str, tier: str) -> Dict[str, str]:
    return {"engine_id": engine_id, "tier": tier}


# ---------------------------------------------------------------------------
# Register metric MVs against the default Prometheus registry. Modules that
# need a private registry (e.g. tests with playground state) construct one
# via :func:`build_private_registry`.
# ---------------------------------------------------------------------------


gpu_seconds_total = Counter(
    "heyavatar_gpu_seconds_total",
    "Cumulative GPU-seconds burned by every finished render chunk.",
    ["engine_id", "tier"],
)

output_minutes_total = Counter(
    "heyavatar_output_minutes_total",
    "Cumulative minutes of useful avatar video produced.",
    ["engine_id", "tier"],
)

queue_depth = Gauge(
    "heyavatar_queue_depth",
    "Number of pending jobs in the queue right now.",
    ["backend", "tier"],
)

jobs_total = Counter(
    "heyavatar_jobs_total",
    "Cumulative count of jobs that reached a terminal state.",
    ["state", "tier"],
)

inflight_jobs = Gauge(
    "heyavatar_inflight_jobs",
    "Number of jobs currently inside a worker (reserved + rendering).",
    ["engine_id"],
)

worker_state_transitions_total = Counter(
    "heyavatar_worker_state_transitions_total",
    "Cumulative count of worker state changes.",
    ["engine_id", "from_state", "to_state"],
)

request_latency_seconds = Histogram(
    "heyavatar_request_latency_seconds",
    "HTTP request latency at the FastAPI gateway.",
    ["method", "route", "status_class"],  # status_class is 2xx, 4xx, 5xx, etc.
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

render_chunk_latency_seconds = Histogram(
    "heyavatar_render_chunk_latency_seconds",
    "Per-chunk GPU render latency.",
    ["engine_id", "degraded"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

process_info = Gauge(
    "heyavatar_process_info",
    "Static info labels — value is always 1, the labels carry metadata.",
    ["service_name", "process_role"],
)
process_info.labels(service_name="heyavatar", process_role=os.environ.get(
    "HEYAVATAR_PROCESS_ROLE", "unspecified"
)).set(1)


# ---------------------------------------------------------------------------
# Public helpers — typed wrappers so application code never touches the
# Prometheus labels directly (a tempting place to add job_id by accident).
# ---------------------------------------------------------------------------


def record_gpu_seconds(engine_id: str, tier: str, gpu_seconds: float) -> None:
    if gpu_seconds <= 0:
        return
    gpu_seconds_total.labels(**_labels_kwargs(engine_id, tier)).inc(float(gpu_seconds))


def record_output_minutes(engine_id: str, tier: str, minutes: float) -> None:
    if minutes <= 0:
        return
    output_minutes_total.labels(**_labels_kwargs(engine_id, tier)).inc(float(minutes))


def observe_request(method: str, route: str, status_code: int,
                    started_monotonic: float) -> None:
    elapsed = max(0.0, time.monotonic() - started_monotonic)
    status_class = f"{status_code // 100}xx"
    request_latency_seconds.labels(
        method=method, route=route, status_class=status_class
    ).observe(elapsed)


def observe_render_chunk(engine_id: str, degraded: bool,
                        started_monotonic: float) -> None:
    elapsed = max(0.0, time.monotonic() - started_monotonic)
    render_chunk_latency_seconds.labels(
        engine_id=engine_id, degraded=str(bool(degraded)).lower()
    ).observe(elapsed)


def set_inflight(engine_id: str, delta: int) -> None:
    if delta == 0:
        return
    if delta > 0:
        inflight_jobs.labels(engine_id=engine_id).inc(delta)
    else:
        inflight_jobs.labels(engine_id=engine_id).dec(-delta)


def set_queue_depth(backend: str, tier: str, depth: int) -> None:
    queue_depth.labels(backend=backend, tier=tier).set(max(0, depth))


def record_worker_state(engine_id: str,
                        from_state: str, to_state: str) -> None:
    worker_state_transitions_total.labels(
        engine_id=engine_id,
        from_state=from_state,
        to_state=to_state,
    ).inc()


def record_terminal(state: str, tier: str) -> None:
    jobs_total.labels(state=state, tier=tier).inc()


# ---------------------------------------------------------------------------
# Exposition helpers — used by both the FastAPI /metrics route and the
# standalone worker HTTP server.
# ---------------------------------------------------------------------------


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


__all__ = [
    "CONTENT_TYPE_LATEST",
    "ENGINE_IDS",
    "TIERS",
    "JOB_STATES",
    "WORKER_STATES",
    "QUEUE_BACKENDS",
    "HTTP_METHODS",
    "build_private_registry",
    "collect_latest",
    "observe_render_chunk",
    "observe_request",
    "record_gpu_seconds",
    "record_output_minutes",
    "record_terminal",
    "record_worker_state",
    "set_inflight",
    "set_queue_depth",
    "gpu_seconds_total",
    "output_minutes_total",
    "queue_depth",
    "jobs_total",
    "inflight_jobs",
    "worker_state_transitions_total",
    "request_latency_seconds",
    "render_chunk_latency_seconds",
]
