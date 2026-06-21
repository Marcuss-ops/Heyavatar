"""Shared fixtures for observability tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def private_registry():
    """Fresh, empty Prometheus registry so tests don't pollute the global one."""
    from src.observability.metrics.exposition import build_private_registry
    yield build_private_registry()
