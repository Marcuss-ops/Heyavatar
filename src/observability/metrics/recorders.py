"""Typed recorders — application code's preferred entry point.

All functions in this module are *guarded*: non-positive increments
or no-op deltas are silently ignored so callers can pass raw
inputs without scrubbing. The label aliases are also normalised so
accidental high-cardinality values cannot leak into the registry.
"""

from __future__ import annotations

import time

from src.observability.metrics.constants import _labels_kwargs
from src.observability.metrics.instruments import (
    inflight_jobs,
    jobs_total,
    output_minutes_total,
    gpu_seconds_total,
    queue_depth,
    render_chunk_latency_seconds,
    request_latency_seconds,
    worker_state_transitions_total,
)


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
