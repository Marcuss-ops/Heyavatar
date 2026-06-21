"""Test TierRouter.pick_available against a real WorkerPool.

Post Change 3 (slimming plan §5 / ROADMAP.md §1) the router is
collapsed to a single ``standard`` profile — there is no fallback walk
between engines. The tests in this module therefore assert:
  * the standard primary is returned when there is an idle worker, and
  * ``None`` is returned when the standard primary has no idle worker
    (the caller decides to wait, fail, or escalate).
"""

from __future__ import annotations

from pathlib import Path

from contracts.avatar_engine import EngineHealth, EngineState
from src.domain.enums import EngineId, Tier
from src.scheduler.routing.router import STANDARD_PROFILE, TierRouter
from src.scheduler.routing.worker_pool import WorkerPool, WorkerRecord

REGISTRY_PATH = Path("registry/models.yaml")


def _make_pool(records: list[WorkerRecord]) -> WorkerPool:
    pool = WorkerPool()
    for record in records:
        pool.register(record)
    return pool


def test_pick_available_returns_standard_primary_when_idle():
    """Pick the standard primary when an idle worker for it exists.

    The MVP standard primary is ``musetalk-v1`` (per the registry).
    """
    router = TierRouter(registry_path=REGISTRY_PATH)
    pool = _make_pool([
        WorkerRecord(
            worker_id="primary",
            engine_id=EngineId.MUSE_TALK,
            vram_total_mb=4096,
            health=EngineHealth(
                engine_id=EngineId.MUSE_TALK, state=EngineState.IDLE
            ),
        ),
    ])
    chosen = router.pick_available(Tier.EXPRESS, pool)
    assert chosen == "musetalk-v1"


def test_pick_available_returns_none_when_standard_primary_busy():
    """Frozen fallback walk: when the standard primary has no idle
    worker the router returns ``None`` instead of falling back to
    a secondary engine.

    Pre-Change-3 this fell back to ``liveportrait-human-v1``. Now
    that the fallback walk is removed per
    ``docs/REPOSITORY_SLIMMING_PLAN.md`` §5 the caller is expected
    to either wait, fail, or escalate the request.
    """
    router = TierRouter(registry_path=REGISTRY_PATH)
    pool = _make_pool([
        WorkerRecord(
            worker_id="busy-primary",
            engine_id=EngineId.MUSE_TALK,
            vram_total_mb=4096,
            in_flight=1,
            health=EngineHealth(
                engine_id=EngineId.MUSE_TALK, state=EngineState.RENDERING
            ),
        ),
        # The liveportrait worker is now intentionally irrelevant:
        # the router must NOT walk to it.
        WorkerRecord(
            worker_id="idle-fallback",
            engine_id=EngineId.LIVE_PORTRAIT,
            vram_total_mb=6144,
            health=EngineHealth(
                engine_id=EngineId.LIVE_PORTRAIT, state=EngineState.IDLE
            ),
        ),
    ])
    chosen = router.pick_available(Tier.EXPRESS, pool)
    assert chosen is None


def test_pick_available_returns_none_when_no_capacity():
    """Empty worker pool yields ``None``."""
    router = TierRouter(registry_path=REGISTRY_PATH)
    pool = _make_pool([])
    chosen = router.pick_available(Tier.EXPRESS, pool)
    assert chosen is None


def test_list_routes_returns_only_standard_profile():
    """``list_routes`` exposes a single row, by design.

    Pre-Change-3 this returned three entries (express / studio /
    premium) plus their fallbacks. Post-Change-3 the only profile is
    ``standard`` and the fallback tuple is empty.
    """
    router = TierRouter(registry_path=REGISTRY_PATH)
    routes = router.list_routes()
    assert routes == [(STANDARD_PROFILE, "musetalk-v1", ())]


def test_for_tier_collapses_all_tiers_to_standard():
    """``for_tier`` returns the same decision regardless of input tier.

    The :class:`Tier` enum keeps historical members for backwards
    API compatibility, but every value resolves to the standard
    profile per Change 3 / ROADMAP.md §1.
    """
    router = TierRouter(registry_path=REGISTRY_PATH)
    decisions = {tier: router.for_tier(tier).primary for tier in Tier}
    assert len(set(decisions.values())) == 1
    assert all(v == "musetalk-v1" for v in decisions.values())
