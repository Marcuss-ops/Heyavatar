"""Tier router — chooses which engine handles a given render request.

The design document shows three quality tiers (express / studio / premium)
and a portfolio of engines. The router takes a tier and a WorkerPool and
returns the engine id best suited to handle the next request.

For v1 the routing is static: every tier has a primary engine plus a
list of fallbacks, declared in :mod:`registry.models_yaml`. Dynamic
routing — using queue depth, VRAM availability, and historical latency —
will be layered on top via a scoring function without changing the
public interface.
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


@dataclass(slots=True, frozen=True)
class RoutingDecision:
    primary: str
    fallbacks: Tuple[str, ...] = ()


@dataclass(slots=True)
class TierRouter:
    registry_path: Path = DEFAULT_REGISTRY

    _rules: Dict[str, RoutingDecision] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.reload()

    def reload(self) -> None:
        rules: Dict[str, RoutingDecision] = {}
        if not self.registry_path.is_file():
            self._rules = rules
            return
        with self.registry_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        tiers = data.get("tiers", {}) or {}
        for tier_name, payload in tiers.items():
            engine = payload.get("engine")
            if not engine:
                continue
            fallbacks = tuple(payload.get("fallbacks", ()) or ())
            rules[tier_name.lower()] = RoutingDecision(primary=engine, fallbacks=fallbacks)
        self._rules = rules

    def for_tier(self, tier: Tier) -> RoutingDecision:
        decision = self._rules.get(tier.value)
        if decision is None:
            raise LookupError(
                f"No routing rule for tier '{tier.value}' in {self.registry_path}."
                " Make sure registry/models.yaml contains a `tiers:` section."
            )
        return decision

    def pick_available(
        self,
        tier: Tier,
        pool: WorkerPool,
    ) -> Optional[str]:
        """Pick the best engine available right now for ``tier``.

        Walks the primary engine first, then fallbacks, and returns the
        first engine id with at least one idle worker in the pool.
        Returns ``None`` if no worker is available — the caller can then
        decide to wait, fail, or escalate to a higher tier.
        """
        decision = self.for_tier(tier)
        candidates = [decision.primary, *decision.fallbacks]
        for engine_id in candidates:
            try:
                eid = EngineId.from_string(engine_id)
            except ValueError:
                continue
            if pool.capacity_for(eid) > 0:
                return engine_id
        return None

    def list_routes(self) -> List[Tuple[str, str, Tuple[str, ...]]]:
        return [
            (tier, dec.primary, dec.fallbacks)
            for tier, dec in sorted(self._rules.items())
        ]
