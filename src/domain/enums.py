"""Enumerations for tier and engine identity."""

from __future__ import annotations

import enum


class Tier(str, enum.Enum):
    """Quality / cost tier chosen by the client."""

    EXPRESS = "express"
    STUDIO = "studio"
    PREMIUM = "premium"

    @classmethod
    def from_string(cls, value: str) -> "Tier":
        try:
            return cls(value.lower())
        except ValueError as exc:
            raise ValueError(f"Unknown tier '{value}'. Use one of: {[t.value for t in cls]}") from exc


class EngineId(str, enum.Enum):
    """Stable identifier of an engine referenced by manifests and adapters."""

    LIVE_PORTRAIT = "liveportrait-human-v1"
    MUSE_TALK = "musetalk-v1"
    ECHO_MIMIC = "echomimic-v1"

    @classmethod
    def from_string(cls, value: str) -> "EngineId":
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(
                f"Unknown engine id '{value}'. Use one of: {[e.value for e in cls]}"
            ) from exc
