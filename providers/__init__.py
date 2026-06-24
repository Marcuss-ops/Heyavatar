"""Provider registry: maps engine ids to adapter classes.

The engine_factory used by the GPU worker reads from this module; tests
also iterate over all registered providers to assert mock-mode parity.
"""

from __future__ import annotations

from importlib import import_module
from typing import Dict

from contracts.avatar_engine import AvatarEngine
from src.domain.enums import EngineId

PROVIDERS: Dict[EngineId, str] = {
    EngineId.LIVE_PORTRAIT: "providers.liveportrait.adapter.engine:LivePortraitAdapter",
    EngineId.MUSE_TALK: "providers.musetalk.adapter.engine:MuseTalkAdapter",
    EngineId.ECHO_MIMIC: "providers.echomimic.adapter:EchoMimicAdapter",
}


def get_provider(engine_id: EngineId) -> AvatarEngine:
    """Instantiate the adapter for ``engine_id``."""
    try:
        target = PROVIDERS[engine_id]
    except KeyError as exc:
        raise KeyError(
            f"No provider registered for engine '{engine_id}'. Registered: "
            f"{sorted(p.value for p in PROVIDERS)}"
        ) from exc
    module_name, class_name = target.split(":", 1)
    module = import_module(module_name)
    cls = getattr(module, class_name)
    return cls()


__all__ = ["PROVIDERS", "get_provider"]
