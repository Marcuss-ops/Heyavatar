"""API /metrics and middleware tests.

These tests use :class:`fastapi.testclient.TestClient` to spin up the
real FastAPI app and check that the metrics exposition endpoint is
mounted and the request latency middleware populates a histogram
sample.
"""

from __future__ import annotations

import pytest


def _has_fastapi() -> bool:
    try:
        import fastapi  # noqa: F401
        return True
    except ImportError:
        return False


def _has_prometheus_client() -> bool:
    try:
        import prometheus_client  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_fastapi(), reason="fastapi not installed")
@pytest.mark.skipif(not _has_prometheus_client(), reason="prometheus_client not installed")
def test_metrics_endpoint_exposes_our_metrics() -> None:
    from fastapi.testclient import TestClient
    from api.app import create_app

    app = create_app()
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.text
    assert "heyavatar_" in body


@pytest.mark.skipif(not _has_fastapi(), reason="fastapi not installed")
@pytest.mark.skipif(not _has_prometheus_client(), reason="prometheus_client not installed")
def test_request_latency_histogram_observes_calls() -> None:
    from fastapi.testclient import TestClient
    from api.app import create_app
    from prometheus_client import REGISTRY as _REG

    app = create_app()
    client = TestClient(app)
    client.get("/ping")
    val = _REG.get_sample_value(
        "heyavatar_request_latency_seconds_count",
        {"method": "GET", "route": "/ping", "status_class": "2xx"},
    )
    assert val is not None and val >= 1
