"""Prometheus metrics facade tests.

All tests in this module skip when ``prometheus-client`` isn't
installed so the standard lightweight CI environment (no optional
extras) keeps working. The tests always use the
``private_registry`` fixture so the counters never pollute the
default global Prometheus registry — that's what makes them
runnable alongside any other suite without producing duplicate
metric registration errors.
"""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.skipif(
    __import__("importlib").util.find_spec("prometheus_client") is None,
    reason="prometheus-client optional extra not installed",
)


def test_gpu_seconds_counter_is_registered(private_registry) -> None:
    from prometheus_client import generate_latest
    body = generate_latest(private_registry).decode("utf-8")
    assert "heyavatar_gpu_seconds_total" in body
    assert "heyavatar_output_minutes_total" in body


def test_record_gpu_seconds_increments_labeled_counter(private_registry) -> None:
    """Exercises the helper functions against the private registry."""
    from src.observability.metrics import (
        record_gpu_seconds,
        record_output_minutes,
    )
    record_gpu_seconds("musetalk-v1", "express", 1.5)
    record_gpu_seconds("musetalk-v1", "express", 0.5)
    record_output_minutes("musetalk-v1", "express", 0.5)
    from prometheus_client import generate_latest
    body = generate_latest(private_registry).decode("utf-8")
    # ``heyavatar_gpu_seconds_total{engine_id="musetalk-v1",tier="express"} 2.0``
    assert 'heyavatar_gpu_seconds_total{engine_id="musetalk-v1",tier="express"} 2.0' in body
    assert 'heyavatar_output_minutes_total{engine_id="musetalk-v1",tier="express"} 0.5' in body


def test_zero_or_negative_increments_are_ignored(private_registry) -> None:
    """Counter helpers refuse non-positive deltas so the ratio stays sane."""
    from src.observability.metrics import record_gpu_seconds, record_output_minutes
    record_gpu_seconds("musetalk-v1", "express", 0.0)
    record_gpu_seconds("musetalk-v1", "express", -3.0)
    record_output_minutes("musetalk-v1", "express", -0.5)
    # Without an active Counter child, the metric is not present at
    # all — that's the desired behaviour.
    from prometheus_client import generate_latest
    body = generate_latest(private_registry).decode("utf-8")
    assert "heyavatar_gpu_seconds_total{engine_id=\"musetalk-v1\",tier=\"express\"}" not in body


def test_observe_request_emits_latency_observation(private_registry) -> None:
    import time
    from src.observability.metrics import observe_request
    started = time.monotonic() - 0.1
    observe_request(method="POST", route="/jobs", status_code=202, started_monotonic=started)
    from prometheus_client import generate_latest
    body = generate_latest(private_registry).decode("utf-8")
    # Histograms emit ``_count`` and ``_sum``. We just check that the
    # path label is present.
    assert 'heyavatar_request_latency_seconds_count{method="POST"' in body


def test_collect_latest_returns_text_plain(private_registry) -> None:
    from src.observability.metrics import collect_latest
    body, content_type = collect_latest(private_registry)
    assert content_type.startswith("text/plain")
    assert b"heyavatar_" in body
