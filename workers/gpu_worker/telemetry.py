"""Telemetry helpers for the GPU worker.

* :func:`_start_metrics_server` — Prometheus exposition HTTP endpoint.
* :func:`_bump_inflight` — increment/decrement of the in-flight gauge.
* :func:`_id_from_str` — defensive ``IdentityId`` vs ``RenderJobId``
  constructor (worker accepts both payloads).
* :func:`read_pack_from_archive` — small helper that lets the worker
  re-read a pack it just wrote, without dragging in the storage layer.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.domain.enums import EngineId
from contracts.avatar_engine import EngineHealth


def _start_metrics_server(port: int) -> None:
    """Start a Prometheus exposition HTTP server on ``port``.

    The server is opt-in: ``port <= 0`` short-circuits, and we degrade
    silently if ``prometheus_client`` is not installed (the worker
    still works; it just won't expose ``/metrics``).
    """
    if port <= 0:
        return
    try:
        from prometheus_client import start_http_server
    except ImportError:
        return
    try:
        start_http_server(port)
    except OSError:
        # Port already bound (e.g. tests or dev hot-reload). Silently skip.
        pass


def _bump_inflight(engine_id: str, delta: int) -> None:
    """Update the per-engine in-flight gauge.

    Best-effort: silently ignores errors so a metrics exporter outage
    never kills a render job.
    """
    try:
        from src.observability.metrics.recorders import set_inflight
        set_inflight(engine_id, delta)
    except ImportError:
        pass


def _id_from_str(value: str):
    """Construct the correct id type from a string token.

    Job ids start with ``job-``; identity ids do not. The worker is
    handed opaque strings from the queue payload, so this is the
    dispatcher between the two domain types.
    """
    from src.domain.types import IdentityId, RenderJobId
    if value.startswith("job-"):
        return RenderJobId(value)
    return IdentityId(value)


def read_pack_from_archive(path: Path):
    """Convenience wrapper so the worker can re-read a pack it just wrote."""
    from src.domain.avatar_pack import read_pack
    return read_pack(path)


# ── distributed heartbeat ──────────────────────────────────────────


def _build_health_payload(
    worker_id: str, engine_id: EngineId, health: EngineHealth
) -> dict:
    """Build the JSON-serialisable health snapshot for Redis publish.

    Schema is documented and matched by
    :meth:`WorkerPool.sync_from_redis` on the API side. Field names
    are deliberately short (no nesting beyond the ``health`` subdict)
    so the wire payload compresses well and future fields can be
    added without churning the API-side parser.
    """
    return {
        "worker_id": worker_id,
        "engine_id": engine_id.value,
        "vram_total_mb": 0,
        "health": {
            "state": health.state.value,
            "vram_used_mb": int(getattr(health, "vram_used_mb", 0) or 0),
            "uptime": float(getattr(health, "uptime_seconds", 0.0) or 0.0),
        },
    }


def _publish_health(
    redis_client: object | None,
    worker_id: str,
    engine_id: EngineId,
    health: EngineHealth,
    ttl_seconds: int,
    log: object,
) -> None:
    """Atomic SET-with-TTL publish on ``heyavatar:worker:{id}:health``.

    Errors are swallowed and logged at DEBUG level: a transient Redis
    blip MUST NOT take down a render worker; the next publish tick
    will overwrite the previous value, and the API-side
    WorkerPool.sync_from_redis() will quietly see a stale snapshot.

    The TTL is set so a crashed worker (no follow-up publish)
    automatically disappears from the cluster view within
    ``ttl_seconds``. Default 15s per the production spec; tunable
    via ``HEYAVATAR_WORKER_POOL_HEARTBEAT_TTL``.
    """
    if redis_client is None:
        return
    try:
        payload = _build_health_payload(worker_id, engine_id, health)
        redis_client.set(
            f"heyavatar:worker:{worker_id}:health",
            json.dumps(payload),
            ex=max(1, int(ttl_seconds)),
        )
    except Exception as exc:  # noqa: BLE001 — must never crash the worker
        try:
            log.debug("Worker heartbeat to Redis skipped: %s", exc)
        except Exception:
            pass
