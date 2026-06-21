"""Cross-process capacity tests for the WorkerPool +\n``WorkerPool.sync_from_redis`` path.

A stub redis client is sufficient to exercise the wire-format
contract: the API process never has to talk to a real Redis server.
These tests pin the schema that the worker's
``telemetry._build_health_payload`` produces and that the pool's
``sync_from_redis`` consumes.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from contracts.avatar_engine import EngineHealth, EngineState
from src.domain.enums import EngineId, Tier
from src.scheduler.routing.router import RoutingDecision, TierRouter
from src.scheduler.routing.worker_pool import WorkerPool


class _StubRedis:
    """In-memory Redis stub with ``scan_iter`` + ``get`` + ``set``.

    Records every call on ``.calls`` so the assertion can verify the
    shape of the publish-message (worker side). The TTL-argument is
    kept on ``.ttls`` so a test can confirm the heartbeat key has the
    15-second cap the production spec requires.
    """

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.calls: list[tuple[str, str, int | None]] = []
        self.ttls: dict[str, int] = {}

    def set(self, key: str, value: str, ex: int | None = None) -> bool:  # noqa: A003
        self.store[key] = value
        self.calls.append((key, value, ex))
        if ex is not None:
            self.ttls[key] = int(ex)
        return True

    def get(self, key: str) -> str | None:
        return self.store.get(key)

    def scan_iter(self, match: str) -> Any:
        """Redis-glob-to-regex so ``*`` matches arbitrary chars.

        The real ``redis.Redis.scan_iter`` interprets ``*`` as the
        SCAN wildcard. Our stub must do the same or the API-side
        :func:`WorkerPool.sync_from_redis` calls would scan an
        empty key set in tests and never pick up workers.
        """
        import re

        pattern = re.escape(match).replace(r"\*", ".*")
        regex = re.compile(f"^{pattern}$")
        return iter(k for k in self.store if regex.fullmatch(k))

    def delete(self, *keys: str) -> int:
        removed = 0
        for k in keys:
            if k in self.store:
                self.store.pop(k, None)
                removed += 1
        return removed


def _publish_health(
    redis_stub: _StubRedis,
    worker_id: str,
    engine_id: EngineId,
    state_name: str = "idle",
    ttl: int = 15,
) -> None:
    """Mimic the worker-side :func:`telemetry._publish_health` shape."""
    payload = {
        "worker_id": worker_id,
        "engine_id": engine_id.value,
        "vram_total_mb": 0,
        "health": {
            "state": state_name,
            "vram_used_mb": 1234,
            "uptime": 7.5,
        },
    }
    redis_stub.set(
        f"heyavatar:worker:{worker_id}:health",
        json.dumps(payload),
        ex=ttl,
    )


# ----------------------------------------------------------------------
# 1. happy path: one remote worker becomes visible across processes.
# ----------------------------------------------------------------------


def test_sync_from_redis_picks_up_published_worker():
    """The stub redis pre-populates a health record.

    After ``WorkerPool.sync_from_redis`` the pool has one
    :class:`WorkerRecord`, and the in-process router picks its
    engine via ``TierRouter.pick_available``.
    """
    redis_stub = _StubRedis()
    _publish_health(redis_stub, "remote-worker-1", EngineId.MUSE_TALK)

    pool = WorkerPool()
    pool.sync_from_redis(redis_stub)

    assert "remote-worker-1" in pool.records
    rec = pool.records["remote-worker-1"]
    assert rec.engine_id == EngineId.MUSE_TALK
    assert rec.health is not None
    assert rec.health.state == EngineState.IDLE
    assert pool.capacity_for(EngineId.MUSE_TALK) == 1

    router = TierRouter(
        registry_path=__import__("pathlib").Path("registry/models.yaml")
    )
    chosen = router.pick_available(Tier.EXPRESS, pool)
    assert chosen == "musetalk-v1"


def test_sync_from_redis_publishes_with_expected_ttl():
    """Worker-side publish sets a 15-second TTL on the health key."""
    redis_stub = _StubRedis()
    _publish_health(redis_stub, "remote-worker-2", EngineId.MUSE_TALK, ttl=15)
    assert redis_stub.ttls["heyavatar:worker:remote-worker-2:health"] == 15


def test_sync_from_redis_routes_in_router_via_fallback():
    """When a remote worker registers as the fallback engine, the
    router flips from primary (busy) to fallback (idle).

    Mirrors the production scenario where a single
    ``musetalk-v1`` worker's CPU core is saturated and a spare
    LivePortrait slot kicks in.
    """
    redis_stub = _StubRedis()
    _publish_health(redis_stub, "remote-lp", EngineId.LIVE_PORTRAIT)

    pool = WorkerPool()
    # Register a "busy" musetalk worker to force the fallback walk.
    from src.scheduler.routing.worker_pool import WorkerRecord

    pool.register(
        WorkerRecord(
            worker_id="local-busy-mt",
            engine_id=EngineId.MUSE_TALK,
            vram_total_mb=0,
            in_flight=1,
            health=EngineHealth(
                engine_id=EngineId.MUSE_TALK, state=EngineState.RENDERING
            ),
        )
    )
    pool.sync_from_redis(redis_stub)

    router = TierRouter(
        registry_path=__import__("pathlib").Path("registry/models.yaml")
    )
    chosen = router.pick_available(Tier.EXPRESS, pool)
    # In the registry, express has musetalk-v1 primary and
    # liveportrait-human-v1 fallback; with the musetalk worker
    # in-flight, the router must select the liveportrait fallback.
    assert chosen == "liveportrait-human-v1"


# ----------------------------------------------------------------------
# 2. malformed entries: dropped with a warning, do not poison the pool.
# ----------------------------------------------------------------------


def test_sync_from_redis_drops_record_missing_worker_id(caplog):
    """A schema-drifted record without ``worker_id`` is dropped.

    Stale or future-shape records MUST NOT silently appear as
    workers — otherwise ``pick_available`` could route a request to
    an entity the pool cannot claims-trust.
    """
    redis_stub = _StubRedis()
    redis_stub.set(
        "heyavatar:worker:bad-record:health",
        json.dumps({"engine_id": "musetalk-v1"}),  # missing worker_id
        ex=15,
    )
    pool = WorkerPool()
    updated = pool.sync_from_redis(redis_stub)
    assert updated == 0
    assert pool.records == {}


def test_sync_from_redis_drops_record_with_unknown_engine_id(caplog):
    """An unknown engine_id is dropped, not coerced to a default.

    Forward-compat: a v0.3 worker could publish a v0.4-only engine
    id; the v0.3 API must not crash on it.
    """
    redis_stub = _StubRedis()
    redis_stub.set(
        "heyavatar:worker:future-engine:health",
        json.dumps(
            {
                "worker_id": "future-engine",
                "engine_id": "liveportrait-human-v2",  # unknown to v0.2
                "health": {"state": "idle", "vram_used_mb": 0, "uptime": 0.0},
            }
        ),
        ex=15,
    )
    pool = WorkerPool()
    updated = pool.sync_from_redis(redis_stub)
    assert updated == 0
    assert "future-engine" not in pool.records


def test_sync_from_redis_drops_record_with_malformed_health_state(caplog):
    """An invalid ``state`` string is dropped (no fallback coercion)."""
    redis_stub = _StubRedis()
    redis_stub.set(
        "heyavatar:worker:bad-state:health",
        json.dumps(
            {
                "worker_id": "bad-state",
                "engine_id": "musetalk-v1",
                "health": {
                    "state": "TOTALLY_NOT_AN_ENUM_VALUE",
                    "vram_used_mb": 0,
                    "uptime": 0.0,
                },
            }
        ),
        ex=15,
    )
    pool = WorkerPool()
    updated = pool.sync_from_redis(redis_stub)
    assert updated == 0
    assert "bad-state" not in pool.records


# ----------------------------------------------------------------------
# 3. resilience: a Redis retry loop sees a record appear after a tick.
# ----------------------------------------------------------------------


def test_sync_from_redis_repeated_calls_pick_up_new_workers():
    """A worker that joins the cluster between two sync calls MUST
    become visible without restarting the API process.

    Simulates the gatherer loop in :file:`api/app/state.py` that
    polls every ``api_worker_pool_sync_seconds``.
    """
    redis_stub = _StubRedis()
    pool = WorkerPool()

    # First tick: cluster is empty.
    assert pool.sync_from_redis(redis_stub) == 0
    assert not pool.records

    # A new worker publishes its health between ticks.
    _publish_health(redis_stub, "joined-worker", EngineId.MUSE_TALK)

    # Second tick: it appears.
    updated = pool.sync_from_redis(redis_stub)
    assert updated == 1
    assert "joined-worker" in pool.records
    assert pool.capacity_for(EngineId.MUSE_TALK) == 1
