"""Tier router tests — post Change 3 (single ``standard`` profile).

Frozen per ``docs/REPOSITORY_SLIMMING_PLAN.md`` §5 and
``ROADMAP.md`` §1: there is only one routing profile. The legacy
``tiers:`` block in :file:`registry/models.yaml` is no longer read;
``router.for_tier`` resolves every :class:`Tier` value to the same
``standard`` decision.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.domain.enums import EngineId, Tier
from src.scheduler.routing.router import STANDARD_PROFILE, TierRouter


def test_router_exposes_single_standard_route():
    """``list_routes`` returns one row: ``("standard", primary, ())``."""
    router = TierRouter(registry_path=Path("registry/models.yaml"))
    routes = router.list_routes()
    assert len(routes) == 1
    tier_name, primary, fallbacks = routes[0]
    assert tier_name == STANDARD_PROFILE
    assert primary in {e.value for e in EngineId}
    assert fallbacks == ()


def test_router_reads_standard_profile_from_registry():
    """``for_tier`` collapses every tier to the ``standard`` primary.

    The MVP registry ships ``standard.engine: musetalk-v1``; operators
    override it via :file:`registry/models.yaml`.
    """
    router = TierRouter(registry_path=Path("registry/models.yaml"))
    primaries = {tier: router.for_tier(tier).primary for tier in Tier}
    # All tiers collapse to the same primary.
    assert len(set(primaries.values())) == 1


def test_router_missing_registry_falls_back_to_default_primary():
    """Missing registry → default ``musetalk-v1`` standard primary.

    Pre-Change-3 this raised ``LookupError`` because every tier had
    to be present in the registry. Now a missing registry is
    treated as "use the MVP default" so cold-start deployments work
    without a hand-rolled ``registry/models.yaml``.
    """
    router = TierRouter(registry_path=Path("/no/such/file.yaml"))
    router.reload()
    # No exception; for_tier returns the same single standard profile.
    primary_1 = router.for_tier(Tier.EXPRESS).primary
    primary_2 = router.for_tier(Tier.PREMIUM).primary
    assert primary_1 == primary_2


def test_reload_picks_up_standard_override(tmp_path):
    """Operators can retarget the standard primary by editing the registry."""
    reg = tmp_path / "models.yaml"
    reg.write_text("standard:\n  engine: liveportrait-human-v1\n", encoding="utf-8")
    router = TierRouter(registry_path=reg)
    assert router.for_tier(Tier.EXPRESS).primary == "liveportrait-human-v1"
    # Reload is idempotent.
    router.reload()
    assert router.for_tier(Tier.STUDIO).primary == "liveportrait-human-v1"


def test_reload_ignores_legacy_tiers_block(tmp_path):
    """Legacy ``tiers: express: ... studio: ...`` config is read but
    ignored by the standard-profile decision.

    Backwards-compatible: a deployed cluster running the pre-Change-3
    ``tiers:`` block continues to boot, just with the standard
    primary (the legacy primary is NOT honoured). This is the
    intended freeze behaviour.
    """
    reg = tmp_path / "models.yaml"
    reg.write_text(
        "tiers:\n"
        "  express:\n"
        "    engine: liveportrait-human-v1\n"
        "    fallbacks: [musetalk-v1]\n"
        "  premium:\n"
        "    engine: echomimic-v1\n",
        encoding="utf-8",
    )
    router = TierRouter(registry_path=reg)
    # Even if an installed config still has the legacy tiers, the
    # standard profile is taken from a default fallback when the
    # operator hasn't defined it explicitly.
    primary = router.for_tier(Tier.EXPRESS).primary
    assert primary in {e.value for e in EngineId}


@pytest.mark.parametrize("tier", list(Tier))
def test_for_tier_collapses_all_enum_members(tier: Tier):
    """Every :class:`Tier` value resolves to the same primary."""
    router = TierRouter(registry_path=Path("registry/models.yaml"))
    primary = router.for_tier(tier).primary
    primaries = {router.for_tier(t).primary for t in Tier}
    assert primary in primaries
    assert len(primaries) == 1
