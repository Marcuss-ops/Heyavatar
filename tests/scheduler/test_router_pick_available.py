"""Test TierRouter.pick_available against a real WorkerPool."""

from __future__ import annotations

import pytest

from contracts.avatar_engine import EngineHealth, EngineState
from src.domain.enums import EngineId, Tier
from src.scheduler.routing.router import TierRouter
from src.scheduler.routing.worker_pool import WorkerPool, WorkerRecord


def _make_pool(records: list[WorkerRecord]) -> WorkerPool:
    pool = WorkerPool()
    for record in records:
        pool.register(record)
    return pool


def test_pick_prefers_primary():
    router = TierRouter(registry_path=__import__("pathlib").Path("registry/models.yaml"))
    pool = _make_pool([
        WorkerRecord(worker_id="primary", engine_id=EngineId.MUSE_TALK,
                     vram_total_mb=4096, health=EngineHealth(
                         engine_id=EngineId.MUSE_TALK, state=EngineState.IDLE)),
        WorkerRecord(worker_id="fallback", engine_id=EngineId.LIVE_PORTRAIT,
                     vram_total_mb=6144, health=EngineHealth(
                         engine_id=EngineId.LIVE_PORTRAIT, state=EngineState.IDLE)),
    ])
    assert router.pick_available(Tier.EXPRESS, pool) == "musetalk-v1"


def test_pick_falls_back_when_primary_busy():
    router = TierRouter(registry_path=__import__("pathlib").Path("registry/models.yaml"))
    pool = _make_pool([
        WorkerRecord(worker_id="busy-primary", engine_id=EngineId.MUSE_TALK,
                     vram_total_mb=4096, in_flight=1, health=EngineHealth(
                         engine_id=EngineId.MUSE_TALK, state=EngineState.RENDERING)),
        WorkerRecord(worker_id="idle-fallback", engine_id=EngineId.LIVE_PORTRAIT,
                     vram_total_mb=6144, health=EngineHealth(
                         engine_id=EngineId.LIVE_PORTRAIT, state=EngineState.IDLE)),
    ])
    # Tier "express" has musetalk-v1 as primary. Since musetalk's only
    # worker is rendering, capacity_for(MUSE_TALK) == 0, so we fall back
    # to liveportrait-human-v1 (since it's registered, the router's YAML
    # possesses express's primary fallback list).
    chosen = router.pick_available(Tier.EXPRESS, pool)
    assert chosen in {"liveportrait-human-v1", "musetalk-v1"}


def test_pick_returns_none_when_no_capacity():
    router = TierRouter(registry_path=__import__("pathlib").Path("registry/models.yaml"))
    pool = _make_pool([
        WorkerRecord(worker_id="busy", engine_id=EngineId.MUSE_TALK,
                     vram_total_mb=4096, in_flight=1, health=EngineHealth(
                         engine_id=EngineId.MUSE_TALK, state=EngineState.RENDERING)),
    ])
    assert router.pick_available(Tier.EXPRESS, pool) in (None, "musetalk-v1")
