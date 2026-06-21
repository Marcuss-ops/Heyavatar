"""Unit tests for the worker-side distributed heartbeat.

Covers :func:`workers.gpu_worker.telemetry._build_health_payload`,
:func:`workers.gpu_worker.telemetry._publish_health`, and the single
-shot :meth:`workers.gpu_worker.worker.GpuWorker.publish_heartbeat_once`
helper used by tests / ops debugging.

The daemon-thread loop is intentionally NOT unit-tested here \u2014 it is
direct logic on top of ``publish_heartbeat_once`` and the
:settings:`HEYAVATAR_WORKER_HEALTH_PUBLISH_SECONDS` period. We test
the integration end-to-end via the ``requirements.txt``/production
deploy path; here we focus on the wire-format contract.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import contracts.avatar_engine as ce
import pytest

from src.domain.enums import EngineId
from src.domain.types import RenderJobId
from workers.gpu_worker.telemetry import (
    _build_health_payload,
    _publish_health,
)


class _StubRedis:
    """In-memory redis stub that records every SET call."""

    def __init__(self, fail_set: bool = False) -> None:
        self.store: dict[str, str] = {}
        self.calls: list[tuple[str, str, int | None]] = []
        self.fail_set = fail_set

    def set(self, key: str, value: str, ex: int | None = None) -> bool:  # noqa: A003
        if self.fail_set:
            raise RuntimeError("simulated redis outage")
        self.store[key] = value
        self.calls.append((key, value, ex))
        return True

    def delete(self, *keys: str) -> int:
        for k in keys:
            self.store.pop(k, None)
        return 1


class _StubLogger:
    def debug(self, *args: Any, **kwargs: Any) -> None:
        pass

    def info(self, *args: Any, **kwargs: Any) -> None:
        pass

    def warning(self, *args: Any, **kwargs: Any) -> None:
        pass

    def error(self, *args: Any, **kwargs: Any) -> None:
        pass


# ----------------------------------------------------------------------
# _build_health_payload — schema contract test.
# ----------------------------------------------------------------------


def test_build_health_payload_serialises_to_dict_with_canonical_keys():
    health = ce.EngineHealth(
        engine_id=EngineId.MUSE_TALK,
        state=ce.EngineState.RENDERING,
        vram_used_mb=2048,
        uptime_seconds=42.5,
    )
    payload = _build_health_payload("worker-007", EngineId.MUSE_TALK, health)
    assert payload == {
        "worker_id": "worker-007",
        "engine_id": "musetalk-v1",
        "vram_total_mb": 0,
        "health": {
            "state": "rendering",
            "vram_used_mb": 2048,
            "uptime": 42.5,
        },
    }
    # Round-trip JSON-serialisable (proves the wire format).
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert decoded == payload


def test_build_health_payload_normalises_imputed_defaults():
    """Health may come in with missing fields (e.g. degraded states).

    The payload MUST still serialise; the API-side parser should be
    able to read it. Defaults (``0`` vram, ``0.0`` uptime) keep the
    schema stable across engine states.
    """
    health = ce.EngineHealth(engine_id=EngineId.LIVE_PORTRAIT,
                             state=ce.EngineState.DEGRADED)
    payload = _build_health_payload("w-degraded", EngineId.LIVE_PORTRAIT, health)
    assert payload["health"]["state"] == "degraded"
    assert payload["health"]["vram_used_mb"] == 0
    assert payload["health"]["uptime"] == 0.0


# ----------------------------------------------------------------------
# _publish_health — wire contract with the stub redis.
# ----------------------------------------------------------------------


def test_publish_health_sets_canonical_key_with_ttl():
    """The wire shape: ``SET heyavatar:worker:{id}:health <json> EX 15``."""
    redis_stub = _StubRedis()
    health = ce.EngineHealth(
        engine_id=EngineId.MUSE_TALK,
        state=ce.EngineState.IDLE,
        vram_used_mb=128,
        uptime_seconds=10.0,
    )
    _publish_health(redis_stub, "w-publish-1", EngineId.MUSE_TALK, health, 15, _StubLogger())

    assert len(redis_stub.calls) == 1
    key, value, ex = redis_stub.calls[0]
    assert key == "heyavatar:worker:w-publish-1:health"
    assert ex == 15
    decoded = json.loads(value)
    assert decoded["worker_id"] == "w-publish-1"
    assert decoded["engine_id"] == "musetalk-v1"
    assert decoded["health"]["state"] == "idle"


def test_publish_health_clamps_minimum_ttl_to_one():
    """``EX 0`` would let redis immediately expire the key. Clamp to 1.

    Matches the production default of 15 while protecting against a
    misconfigured ``HEYAVATAR_WORKER_POOL_HEARTBEAT_TTL=0`` (which
    would otherwise make the cross-process capacity tracker unable
    to ever observe the worker).
    """
    redis_stub = _StubRedis()
    health = ce.EngineHealth(engine_id=EngineId.MUSE_TALK,
                             state=ce.EngineState.IDLE)
    _publish_health(redis_stub, "w-zero-ttl", EngineId.MUSE_TALK, health, 0, _StubLogger())
    _, _, ex = redis_stub.calls[0]
    assert ex == 1


def test_publish_health_swallows_redis_errors():
    """A transient Redis blip MUST NOT crash the worker.

    The publish MUST log a debug line and return; the next publish
    tick (``worker_health_publish_seconds`` later) retries against a
    fresh client.
    """
    redis_stub = _StubRedis(fail_set=True)
    health = ce.EngineHealth(engine_id=EngineId.MUSE_TALK,
                             state=ce.EngineState.IDLE)
    # No raise \u2014 the helper is total.
    _publish_health(redis_stub, "w-blip", EngineId.MUSE_TALK, health, 15, _StubLogger())


def test_publish_health_no_op_when_redis_client_is_none():
    """``_publish_health`` is called in the no-redis dev path.

    The helper must short-circuit and not crash on ``None``.
    """
    health = ce.EngineHealth(engine_id=EngineId.MUSE_TALK,
                             state=ce.EngineState.IDLE)
    _publish_health(None, "w-noredis", EngineId.MUSE_TALK, health, 15, _StubLogger())


# ----------------------------------------------------------------------
# GpuWorker.publish_heartbeat_once \u2014 single-shot integration.
# ----------------------------------------------------------------------


def test_publish_heartbeat_once_writes_to_injected_client(monkeypatch):
    """The single-shot helper bypasses the background thread and
    publishes once via the injected redis client.

    We monkey-patch ``GpuWorker._load_engine`` to return a stub
    with a deterministic ``health()`` so the wire payload is
    pinned.
    """
    from contracts.job_queue import QueueHandle
    from src.scheduler.queue.null import NullJobQueue
    from src.storage.avatar_packs import AvatarPackRepository
    from workers.gpu_worker.worker import GpuWorker
    from src.core.config import get_settings

    class _StubEngine:
        def __init__(self) -> None:
            self.calls = 0

        def health(self) -> ce.EngineHealth:
            self.calls += 1
            return ce.EngineHealth(
                engine_id=EngineId.LIVE_PORTRAIT,
                state=ce.EngineState.IDLE,
                vram_used_mb=512,
                uptime_seconds=3.0,
            )

    stub_engine = _StubEngine()
    monkeypatch.setattr(
        GpuWorker, "_load_engine", lambda self: stub_engine
    )

    redis_stub = _StubRedis()
    settings = dataclasses.replace(get_settings(), worker_pool_heartbeat_ttl=15)
    worker = GpuWorker(
        engine_id=EngineId.LIVE_PORTRAIT,
        settings=settings,
        pack_repo=AvatarPackRepository(root=settings.pack_dir),
        queue=NullJobQueue(),
        handle=QueueHandle(
            worker_id="gpu-worker-001",
            engine_id="liveportrait-human-v1",
            tier="any",
        ),
        redis_client=redis_stub,
    )

    assert worker.publish_heartbeat_once(redis_client=redis_stub) is True
    assert stub_engine.calls == 1
    assert len(redis_stub.calls) == 1
    key, value, ex = redis_stub.calls[0]
    assert key == "heyavatar:worker:gpu-worker-001:health"
    assert ex == 15
    decoded = json.loads(value)
    assert decoded["worker_id"] == "gpu-worker-001"
    assert decoded["engine_id"] == "liveportrait-human-v1"
    assert decoded["health"]["state"] == "idle"


def test_publish_heartbeat_once_returns_false_when_no_redis_client(monkeypatch):
    """``redis_client is None`` AND no ``settings.redis_url`` resolves.

    The helper must short-circuit and return False (not raise).
    """
    from contracts.job_queue import QueueHandle
    from src.scheduler.queue.null import NullJobQueue
    from src.storage.avatar_packs import AvatarPackRepository
    from src.core.config import get_settings
    from workers.gpu_worker.worker import GpuWorker

    class _StubEngine:
        def health(self) -> ce.EngineHealth:
            return ce.EngineHealth(
                engine_id=EngineId.MUSE_TALK, state=ce.EngineState.IDLE
            )

    monkeypatch.setattr(GpuWorker, "_load_engine", lambda self: _StubEngine())

    settings = dataclasses.replace(
        get_settings(), redis_url=None, worker_pool_heartbeat_ttl=15
    )
    worker = GpuWorker(
        engine_id=EngineId.MUSE_TALK,
        settings=settings,
        pack_repo=AvatarPackRepository(root=settings.pack_dir),
        queue=NullJobQueue(),
        handle=QueueHandle(worker_id="w", engine_id="musetalk-v1", tier="any"),
        redis_client=None,
    )
    assert worker.publish_heartbeat_once(redis_client=None) is False
