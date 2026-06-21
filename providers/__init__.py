"""Provider registry: maps engine ids to adapter classes.

The engine_factory used by the GPU worker reads from this module; tests
also iterate over all registered providers to assert mock-mode parity.
"""

from __future__ import annotations

from typing import Dict, Type

from contracts.avatar_engine import AvatarEngine
from src.domain.enums import EngineId

from providers.echomimic.adapter import EchoMimicAdapter
from providers.liveportrait.adapter.engine import LivePortraitAdapter
from providers.musetalk.adapter.engine import MuseTalkAdapter

PROVIDERS: Dict[EngineId, Type[AvatarEngine]] = {
    EngineId.LIVE_PORTRAIT: LivePortraitAdapter,
    EngineId.MUSE_TALK: MuseTalkAdapter,
    EngineId.ECHO_MIMIC: EchoMimicAdapter,
}


def get_provider(engine_id: EngineId) -> AvatarEngine:
    """Instantiate the adapter for ``engine_id``."""
    try:
        cls = PROVIDERS[engine_id]
    except KeyError as exc:
        raise KeyError(
            f"No provider registered for engine '{engine_id}'. Registered: "
            f"{sorted(p.value for p in PROVIDERS)}"
        ) from exc
    return cls()


__all__ = ["EchoMimicAdapter", "LivePortraitAdapter", "MuseTalkAdapter", "PROVIDERS", "get_provider"]
