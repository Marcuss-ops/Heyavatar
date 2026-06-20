"""Telemetry — GPU-second accounting and cost metrics.

The single most important business metric is **GPU-seconds consumed per
minute of useful avatar video produced**. Everything flows from there.
"""

from __future__ import annotations

import contextlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterator


@dataclass(slots=True)
class TelemetryRecorder:
    """Lightweight counter aggregated in-process; replace with Prometheus later."""

    gpu_seconds_total: float = 0.0
    inference_count: int = 0
    latencies_ms: Dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    per_engine_gpu_seconds: Dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def record(self, gpu_seconds: float, *, engine_id: str) -> None:
        self.gpu_seconds_total += float(gpu_seconds)
        self.per_engine_gpu_seconds[engine_id] += float(gpu_seconds)

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
            "inference_count": self.inference_count,
            "per_engine_gpu_seconds": dict(self.per_engine_gpu_seconds),
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
