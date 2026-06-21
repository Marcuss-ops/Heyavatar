"""Provider registry: maps engine ids to adapter classes.

The engine_factory used by the GPU worker reads from this module; tests
also iterate over all registered providers to assert mock-mode parity.

EchoMimic (engine_id ``echomimic-v1``) is **FROZEN** per
``ROADMAP.md`` §1: the directory ``providers/echomimic/`` is kept on
disk for forward compatibility, the ``EngineId.ECHO_MIMIC`` enum value
is preserved so registry / database identifiers stay valid, but the
adapter is intentionally NOT registered in :data:`PROVIDERS`. Any
attempt to construct an EchoMimic adapter via :func:`get_provider`
raises :class:`KeyError` — preserving "no real adapter path" with a
clean error rather than a silent ``NotImplementedError``.
"""

from __future__ import annotations

from typing import Dict, Type

from contracts.avatar_engine import AvatarEngine
from src.domain.enums import EngineId

from providers.liveportrait.adapter.engine import LivePortraitAdapter
from providers.musetalk.adapter.engine import MuseTalkAdapter

PROVIDERS: Dict[EngineId, Type[AvatarEngine]] = {
    EngineId.LIVE_PORTRAIT: LivePortraitAdapter,
    EngineId.MUSE_TALK: MuseTalkAdapter,
    # EngineId.ECHO_MIMIC is FROZEN per Change 3 / ROADMAP §1 — not
    # registered. The value is preserved on the ``EngineId`` enum so
    # registry / database identifiers stay valid; ``get_provider`` raises
    # ``KeyError`` with a frozen-engine message when callers ask for it.
}


def get_provider(engine_id: EngineId) -> AvatarEngine:
    """Instantiate the adapter for ``engine_id``.

    Raises :class:`KeyError` for engine ids that are not registered,
    including frozen engines such as ``ECHO_MIMIC`` (preserved in the
    :class:`src.domain.enums.EngineId` enum but deliberately omitted
    from :data:`PROVIDERS` per ``ROADMAP.md`` §1).
    """
    try:
        cls = PROVIDERS[engine_id]
    except KeyError as exc:
        registered = sorted(p.value for p in PROVIDERS)
        if engine_id == EngineId.ECHO_MIMIC:
            raise KeyError(
                f"Engine '{engine_id}' is frozen (Change 3 of slimming plan, "
                f"ROADMAP §1). Registered active engines: {registered}."
            ) from exc
        raise KeyError(
            f"No provider registered for engine '{engine_id}'. "
            f"Registered: {registered}"
        ) from exc
    return cls()


__all__ = [
    "LivePortraitAdapter",
    "MuseTalkAdapter",
    "PROVIDERS",
    "get_provider",
]
