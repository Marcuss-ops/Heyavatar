"""Telemetry — GPU-second accounting and cost metrics.

Two parallel paths:

* a process-local in-memory dataclass (``TelemetryRecorder``), kept
  for back-compat and tests that don't want to talk to a Prometheus
  collector registry.
* the Prometheus Counter ``heyavatar_gpu_seconds_total`` and
  ``heyavatar_output_minutes_total`` registered in
  :mod:`src.observability.metrics`. The dashboard headline metric
  (``gpu_seconds_per_minute_of_output``) is the ratio of their
  ``rate()`` values in Grafana.

Workers SHOULD publish with explicit tier via
:meth:`TelemetryRecorder.publish_metrics` so the dashboard
``rate(...) by (engine_id, tier)`` panels are populated correctly.
"""

from __future__ import annotations

import contextlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional


@dataclass(slots=True)
class TelemetryRecorder:
    """Lightweight counter aggregated in-process; Prometheus is the wire."""

    gpu_seconds_total: float = 0.0
    inference_count: int = 0
    latencies_ms: Dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    per_engine_gpu_seconds: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    output_minutes_total: float = 0.0
    per_engine_output_minutes: Dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def record(self, gpu_seconds: float, *, engine_id: str) -> None:
        """Back-compat path: same signature as the pre-observability build.

        Promotes into the ``express`` tier label by default; callers
        that know the request tier should call :meth:`publish_metrics`
        directly so the dashboard gets the correct labels.
        """
        self.gpu_seconds_total += float(gpu_seconds)
        self.per_engine_gpu_seconds[engine_id] += float(gpu_seconds)
        self.publish_metrics(engine_id=engine_id, tier="express",
                             gpu_seconds=float(gpu_seconds), output_minutes=0.0)

    def publish_metrics(
        self,
        *,
        engine_id: str,
        tier: str,
        gpu_seconds: float,
        output_minutes: float,
    ) -> None:
        """Publish the per-chunk economics into both the dataclass and
        the Prometheus Counter side-channel.

        The side-channel is best-effort: if ``prometheus-client`` is
        not installed (e.g. a CPU-only development environment), the
        dataclass still absorbs the value so /healthz can show totals.
        """
        if gpu_seconds > 0:
            self.gpu_seconds_total += float(gpu_seconds)
            self.per_engine_gpu_seconds[engine_id] += float(gpu_seconds)
        if output_minutes > 0:
            self.output_minutes_total += float(output_minutes)
            self.per_engine_output_minutes[engine_id] += float(output_minutes)
        try:
            from src.observability.metrics import (
                record_gpu_seconds,
                record_output_minutes,
            )
            if gpu_seconds > 0:
                record_gpu_seconds(engine_id, tier, float(gpu_seconds))
            if output_minutes > 0:
                record_output_minutes(engine_id, tier, float(output_minutes))
        except ImportError:
            pass

    @contextlib.contextmanager
    def span(self, name: str, **tags) -> Iterator[None]:
        """Time-context that records latency to the named span."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            key = f"{name}:{','.join(f'{k}={v}' for k, v in sorted(tags.items()))}"
            self.latencies_ms[key].append(elapsed_ms)

    def snapshot(self) -> dict:
        return {
            "gpu_seconds_total": round(self.gpu_seconds_total, 4),
            "output_minutes_total": round(self.output_minutes_total, 4),
            "inference_count": self.inference_count,
            "per_engine_gpu_seconds": dict(self.per_engine_gpu_seconds),
            "per_engine_output_minutes": dict(self.per_engine_output_minutes),
            "avg_latency_ms_by_span": {
                k: round(sum(values) / len(values), 2) for k, values in self.latencies_ms.items()
            },
            "avg_fps_last_job": self._estimate_fps(),
        }

    def _estimate_fps(self) -> float:
        if not self.latencies_ms:
            return 0.0
        recent = max(self.latencies_ms.values(), key=len)
        if not recent:
            return 0.0
        avg_ms_per_chunk = sum(recent[-10:]) / min(len(recent), 10)
        # 4-second chunks at 25 fps → 100 frames per chunk; FPS approximation:
        return 0.0 if avg_ms_per_chunk == 0 else round(100 / avg_ms_per_chunk * 1000.0, 2)
