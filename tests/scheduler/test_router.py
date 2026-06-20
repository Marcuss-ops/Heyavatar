"""Tier router tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.domain.enums import Tier
from src.scheduler.router import TierRouter


def test_router_reads_registry_yaml():
    router = TierRouter(registry_path=Path("registry/models.yaml"))
    express = router.for_tier(Tier.EXPRESS)
    assert express.primary
    premium = router.for_tier(Tier.PREMIUM)
    assert premium.primary
    routes = router.list_routes()
    assert ("express", express.primary, express.fallbacks) in routes


def test_router_missing_registry_raises(tmp_path):
    router = TierRouter(registry_path=tmp_path / "no-such-file.yaml")
    router.reload()  # does not crash
    with pytest.raises(LookupError):
        router.for_tier(Tier.EXPRESS)
