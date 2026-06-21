"""Prometheus ``Counter`` / ``Gauge`` / ``Histogram`` declarations.

These instruments are bound to the default Prometheus collector
registry at import time. Application code should call the typed
recorders in :mod:`recorders` rather than touching these directly,
to avoid accidentally introducing high-cardinality labels.
"""

from __future__ import annotations

import os

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
)
from prometheus_client import REGISTRY as _DEFAULT_REGISTRY


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
