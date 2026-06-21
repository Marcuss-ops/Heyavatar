"""In-process capacity tracking tests for :class:`WorkerPool`.

These verify the lifecycle wiring (register / heartbeat / mark_in_flight /
unregister) and demonstrate that
:meth:`TierRouter.pick_available` walks fallbacks correctly. We avoid
reading the on-disk ``registry/models.yaml`` so the tests are stable
across registry edits.
"""

from __future__ import annotations

import contracts.avatar_engine as ce
import pytest

from src.domain.enums import EngineId, Tier
from src.scheduler.routing.router import RoutingDecision, TierRouter
from src.scheduler.routing.worker_pool import WorkerPool, WorkerRecord


def _record(worker_id: str, engine: EngineId, state=ce.EngineState.IDLE) -> WorkerRecord:
    return WorkerRecord(
        worker_id=worker_id,
        engine_id=engine,
        vram_total_mb=8192,
        health=ce.EngineHealth(engine_id=engine, state=state),
    )


def test_register_and_idle_capacity():
    pool = WorkerPool()
    pool.register(_record("w-1", EngineId.LIVE_PORTRAIT))
    assert pool.capacity_for(EngineId.LIVE_PORTRAIT) == 1


def test_mark_in_flight_consumes_capacity():
    pool = WorkerPool()
    pool.register(_record("w-1", EngineId.LIVE_PORTRAIT))
    pool.mark_in_flight("w-1", +1)
    assert pool.capacity_for(EngineId.LIVE_PORTRAIT) == 0
    pool.mark_in_flight("w-1", -1)
    assert pool.capacity_for(EngineId.LIVE_PORTRAIT) == 1


def test_mark_in_flight_clamps_at_zero():
    pool = WorkerPool()
    pool.register(_record("w-1", EngineId.LIVE_PORTRAIT))
    pool.mark_in_flight("w-1", -5)  # counter must not go negative
    assert pool.records["w-1"].in_flight == 0


def test_mark_in_flight_unknown_worker_is_noop():
    pool = WorkerPool()
    pool.mark_in_flight("ghost", +1)
    assert "ghost" not in pool.records


def test_heartbeat_updates_only_health():
    pool = WorkerPool()
    pool.register(_record("w-1", EngineId.LIVE_PORTRAIT))
    pool.heartbeat(
        "w-1",
        health=ce.EngineHealth(
            engine_id=EngineId.LIVE_PORTRAIT,
            state=ce.EngineState.RENDERING,
        ),
    )
    assert pool.records["w-1"].health.state == ce.EngineState.RENDERING


def test_unregister_drops_record():
    pool = WorkerPool()
    pool.register(_record("w-1", EngineId.LIVE_PORTRAIT))
    pool.unregister("w-1")
    assert "w-1" not in pool.records


def test_unregister_unknown_worker_does_not_raise():
    pool = WorkerPool()
    pool.unregister("ghost")  # idempotent


def test_pick_available_walks_primary_then_fallback():
    """**FROZEN per Change 3 / ROADMAP.md §1.** The router no longer
    walks a fallback list. Pre-Change-3 this test asserted that a
    studio request whose primary (liveportrait-human-v1) was DEGRADED
    fell back to ``musetalk-v1``. Now that the fallback walk is
    removed the router returns ``None`` if the standard primary has
    no idle worker, regardless of other engines' capacity.

    We pin the new contracted behaviour here: the liveportrait worker
    is DEGRADED, no musetalk worker is registered, and
    ``pick_available`` must return ``None``.
    """
    pool = WorkerPool()
    pool.register(_record("w-lp", EngineId.LIVE_PORTRAIT,
                          state=ce.EngineState.DEGRADED))
    # Deliberately do NOT register a musetalk worker. Pre-Change-3 the
    # router would still pick musetalk as fallback; post-Change-3 it
    # returns None because the standard primary (musetalk) has zero
    # capacity.
    router = TierRouter(registry_path=__import__(
        "pathlib", fromlist=["Path"]
    ).Path("does-not-exist.yaml"))
    # No route injection: post-Change-3 the router exposes a single
    # standard profile (`musetalk-v1`) and ignores `_rules` for the
    # pickup. We assert on the behaviour exactly.
    assert router.pick_available(Tier.STUDIO, pool) is None
    assert router.pick_available(Tier.EXPRESS, pool) is None


def test_pick_available_returns_none_when_no_idle_worker():
    """**Frozen behaviour.** Single standard primary, returns ``None``
    when no idle worker is registered for it.
    """
    pool = WorkerPool()
    pool.register(_record("w-lp", EngineId.LIVE_PORTRAIT,
                          state=ce.EngineState.DEGRADED))
    router = TierRouter(registry_path=__import__(
        "pathlib", fromlist=["Path"]
    ).Path("does-not-exist.yaml"))
    # The standard primary is musetalk-v1; with no musetalk worker
    # registered the router returns None.
    assert router.pick_available(Tier.EXPRESS, pool) is None
    # list_routes exposes a single row regardless of pool contents.
    routes = router.list_routes()
    assert len(routes) == 1
    assert routes[0][0] == "standard"


def test_sync_from_redis_returns_zero_when_no_client():
    pool = WorkerPool()
    assert pool.sync_from_redis(None) == 0


def test_sync_from_redis_skips_unparseable_value(caplog):
    """A record whose JSON body cannot be decoded must be skipped.

    This protects the cluster view from being poisoned by a
    truncated write or a schema drift in an old worker version
    still publishing a record whose top-level structure changed.
    """
    import json as _json
    import sys
    # Local stub class: avoid an external import that might be unavailable.
    class _CorruptRedis:
        def __init__(self):
            self.store = {"heyavatar:worker:c:health": "{not json"}
            self.yielded = []

        def scan_iter(self, match):
            prefix = match.rstrip("*")
            self.yielded.extend(k for k in self.store if k.startswith(prefix))
            return iter(self.yielded)

        def get(self, key):
            return self.store.get(key)

    pool = WorkerPool()
    assert pool.sync_from_redis(_CorruptRedis()) == 0
    assert pool.records == {}
