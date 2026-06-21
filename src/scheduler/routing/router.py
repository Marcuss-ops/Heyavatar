"""Tier router — collapsed to one profile per Change 3 of the slimming plan.

The slimming plan (see ``docs/REPOSITORY_SLIMMING_PLAN.md`` §5 and
``ROADMAP.md`` §1) freezes multi-tier routing: the MVP exposes a single
``standard`` profile (LivePortrait+Musetalk, 25 FPS, 256×256 face ROI,
H.264 output, prerecorded body template). The router keeps the public
API (``for_tier``, ``pick_available``, ``list_routes``) so existing
callers and tests do not need to change; the engine returned is
always the registry's ``standard.primary`` regardless of the
:class:`src.domain.enums.Tier` value the caller passes. Future
``EXPRESS``-class enum members and any out-of-tree queue payloads
therefore degrade gracefully to ``standard`` rather than blowing up.

No fallback walk, no dynamic capacity scoring, no per-tier quality tier
distinction. The router is a thin mapping: ``tier → "standard.primary"``.

The legacy ``tiers:`` block in :file:`registry/models.yaml` is still
read (for backwards compatibility with installed configs) but every
legacy tier's decision is ignored — only the ``standard`` block is
honored. Operators wanting to retarget the standard profile edit
``registry/models.yaml::standard.engine`` and restart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from src.domain.enums import EngineId, Tier

# Sibling submodule — keeps the intra-package coupling explicit.
from .worker_pool import WorkerPool


DEFAULT_REGISTRY = Path("registry/models.yaml")
STANDARD_PROFILE = "standard"


@dataclass(slots=True, frozen=True)
class RoutingDecision:
    primary: str
    fallbacks: Tuple[str, ...] = ()


@dataclass(slots=True)
class TierRouter:
    registry_path: Path = DEFAULT_REGISTRY

    _standard: RoutingDecision = field(
        init=False, repr=False, default_factory=lambda: RoutingDecision(
            primary=EngineId.MUSE_TALK.value,
        )
    )

    def __post_init__(self) -> None:
        self.reload()

    def reload(self) -> None:
        """Refresh the standard-profile decision from ``registry/models.yaml``.

        On startup ``standard.primary`` defaults to ``musetalk-v1``. If
        ``registry/models.yaml`` defines a ``standard`` block the
        operator override takes effect. The legacy ``tiers:`` block is
        read but intentionally ignored.
        """
        primary_default = EngineId.MUSE_TALK.value
        # Enforce that the primary string is one of the registered
        # enum values; fall back to ``MUSE_TALK`` if a stray
        # ``standard.primary`` reference points to a frozen engine id.
        if not self.registry_path.is_file():
            self._standard = RoutingDecision(primary=primary_default)
            return
        with self.registry_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        std = data.get(STANDARD_PROFILE) or {}
        primary = std.get("engine") if isinstance(std, dict) else None
        try:
            eid = EngineId.from_string(primary) if primary else None
        except ValueError:
            eid = None
        if eid is None:
            eid = EngineId.MUSE_TALK
        self._standard = RoutingDecision(primary=eid.value)

    def for_tier(self, tier: Tier) -> RoutingDecision:
        """Return the (only) routing decision for ``tier``.

        All tier values resolve to the collapsed ``standard`` profile
        per Change 3 / ``ROADMAP.md`` §1. The :class:`src.domain.enums.Tier`
        enum keeps historical values for backwards compatibility.
        """
        # Touch ``tier`` so callers see a symmetric API. The actual
        # return is always the standard profile.
        del tier  # noqa: ARG001 — intentional; tier is ignored
        return self._standard

    def pick_available(
        self,
        tier: Tier,
        pool: WorkerPool,
    ) -> Optional[str]:
        """Pick the engine available right now for ``tier``.

        Frozen-tier fallback walk removed (Change 3): the router always
        returns the standard profile's primary engine and only the
        primary. If there's no idle worker for it the router returns
        ``None`` so the caller can wait, fail, or escalate. The
        capacity-aware logic is in :class:`WorkerPool`.
        """
        decision = self.for_tier(tier)
        try:
            eid = EngineId.from_string(decision.primary)
        except ValueError:
            return None
        if pool.capacity_for(eid) > 0:
            return decision.primary
        return None

    def list_routes(self) -> List[Tuple[str, str, Tuple[str, ...]]]:
        """Return the canonical route table — exactly one entry.

        ``("standard", primary, ())`` — no fallbacks because the
        multi-tier fallback walk is frozen.
        """
        dec = self._standard
        return [(STANDARD_PROFILE, dec.primary, dec.fallbacks)]
