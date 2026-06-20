"""AvatarEngine contract — the stable interface every provider implements.

The contract is deliberately narrow so a LivePortrait, MuseTalk, EchoMimic,
or a future in-house engine can be swapped without touching the rest of
the platform. Conversely, no application code imports the provider
modules directly; it imports :class:`AvatarEngine`.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict

from src.domain.types import (
    AvatarIdentityHandle,
    RenderChunkRequest,
    RenderChunkResult,
)
from src.domain.enums import EngineId


class EngineState(str, Enum):
    """Cheap state machine surfaced by :meth:`AvatarEngine.health`."""

    LOADING = "loading"
    IDLE = "idle"
    RENDERING = "rendering"
    DEGRADED = "degraded"
    UNLOADED = "unloaded"
    ERROR = "error"


@dataclass(slots=True, frozen=True)
class EngineHealth:
    """Snapshot of an engine's runtime condition."""

    engine_id: EngineId
    state: EngineState
    vram_used_mb: int = 0
    last_inference_latency_ms: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    uptime_seconds: float = 0.0
    mock_mode: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine_id": self.engine_id.value,
            "state": self.state.value,
            "vram_used_mb": self.vram_used_mb,
            "last_inference_latency_ms": self.last_inference_latency_ms,
            "metrics": dict(self.metrics),
            "uptime_seconds": self.uptime_seconds,
            "mock_mode": self.mock_mode,
        }


class AvatarEngine(abc.ABC):
    """Stable contract every talking-head rendering engine implements.

    Implementations are expected to be persistent (loaded once, reused
    across many requests) for cost reasons. VRAM is reclaimed by
    disposing the worker process — see ``workers/gpu_worker.py``.
    """

    engine_id: ClassVar[EngineId]

    def __init__(self) -> None:
        self._loaded_at: float | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def load(self) -> None:
        """Load weights, warm up CUDA kernels, advance state to IDLE."""

    @abc.abstractmethod
    def unload(self) -> None:
        """Release resources; the engine must be re-loadable after this."""

    # ------------------------------------------------------------------
    # Per-identity preparation — pays off for many videos per avatar
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def prepare_identity(self, source_image: Path) -> AvatarIdentityHandle:
        """Run the on-boarding pipeline that produces an Avatar Pack."""

    # ------------------------------------------------------------------
    # Per-chunk rendering — the hot path
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def render_chunk(
        self,
        request: RenderChunkRequest,
        identity: AvatarIdentityHandle,
    ) -> RenderChunkResult:
        """Render a short audio-driven chunk and return raw frame metadata."""

    # ------------------------------------------------------------------
    # Lifecycle inspection
    # ------------------------------------------------------------------
    def health(self) -> EngineHealth:
        """Return current health. Implementations should override lightly."""
        uptime = (time.monotonic() - self._loaded_at) if self._loaded_at else 0.0
        return EngineHealth(engine_id=self.engine_id, state=EngineState.IDLE, uptime_seconds=uptime)

    def mark_loaded(self) -> None:
        self._loaded_at = time.monotonic()


__all__ = ["AvatarEngine", "EngineState", "EngineHealth"]
