"""Worker pool — capacity tracker and heartbeat.

The pool tracks which engines are alive in which worker process. The
scheduler never owns the GPU; the worker does. The pool is responsible
for restart-on-VRAM-pressure decisions (``unload + load`` is not enough
when VRAM is fragmented — see design notes in README).
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from contracts.avatar_engine import EngineHealth, EngineState
from src.domain.enums import EngineId


@dataclass(slots=True)
class WorkerRecord:
    """Live state of a worker process."""

    worker_id: str
    engine_id: EngineId
    vram_total_mb: int
    last_heartbeat: float = field(default_factory=time.monotonic)
    health: EngineHealth | None = None
    in_flight: int = 0


@dataclass(slots=True)
class WorkerPool:
    """In-process registry of GPU workers (one record per live worker)."""

    records: Dict[str, WorkerRecord] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def register(self, record: WorkerRecord) -> None:
        with self._lock:
            self.records[record.worker_id] = record

    def heartbeat(self, worker_id: str, *, health: Optional[EngineHealth] = None) -> None:
        with self._lock:
            r = self.records.get(worker_id)
            if r is None:
                return
            self.records[worker_id] = WorkerRecord(
                worker_id=r.worker_id,
                engine_id=r.engine_id,
                vram_total_mb=r.vram_total_mb,
                last_heartbeat=time.monotonic(),
                health=health or r.health,
                in_flight=r.in_flight,
            )

    def find_idle(self, engine_id: EngineId) -> List[WorkerRecord]:
        with self._lock:
            return [r for r in self.records.values()
                    if r.engine_id == engine_id
                    and r.in_flight == 0
                    and (r.health is None
                         or r.health.state == EngineState.IDLE)]

    def capacity_for(self, engine_id: EngineId) -> int:
        return len(self.find_idle(engine_id))

    def unregister(self, worker_id: str) -> None:
        """Remove a worker record from the pool.

        Idempotent: unknown worker ids are silently ignored so callers
        can deregister on the cleanup path without first checking.
        """
        with self._lock:
            self.records.pop(worker_id, None)

    def mark_in_flight(self, worker_id: str, delta: int) -> None:
        """Increment or decrement the in-flight job count for ``worker_id``.

        No-op if the worker isn't registered (does NOT implicitly
        register, to keep caller-side wiring explicit). The counter is
        clamped at 0 — a stray ``-1`` cannot produce a negative
        in-flight count.
        """
        with self._lock:
            r = self.records.get(worker_id)
            if r is None:
                return
            self.records[worker_id] = WorkerRecord(
                worker_id=r.worker_id,
                engine_id=r.engine_id,
                vram_total_mb=r.vram_total_mb,
                last_heartbeat=r.last_heartbeat,
                health=r.health,
                in_flight=max(0, r.in_flight + delta),
            )

    def sync_from_redis(self, redis_client) -> int:
        """Pull worker records published by distributed GPU processes.

        Producers (one per worker process) write to
        ``heyavatar:worker:{worker_id}:health`` with a TTL so crashed
        workers disappear automatically. This method ``SCAN``s the
        pattern and calls :meth:`register` / :meth:`heartbeat` for each
        live record it sees.

        Returns the number of records updated (0 if no records yet).

        Degraded gracefully when ``redis_client`` is None (no-op).
        Schema-drifted records are dropped **with a WARNING log** so an
        operator sees a silent loss of capacity in ``journalctl``.
        """
        if redis_client is None:
            return 0
        from src.core.logging import get_logger
        log = get_logger(__name__)
        try:
            updated = 0
            for key in redis_client.scan_iter(match="heyavatar:worker:*:health"):
                raw = redis_client.get(key)
                if not raw:
                    continue
                import json
                try:
                    payload = json.loads(raw)
                except (ValueError, OSError) as exc:
                    log.warning("WorkerPool.sync_from_redis: dropping record %s (%s)", key, exc)
                    continue
                worker_id = payload.get("worker_id")
                engine_id_str = payload.get("engine_id")
                if not worker_id or not engine_id_str:
                    log.warning("WorkerPool.sync_from_redis: %s missing worker_id/engine_id", key)
                    continue
                try:
                    eid = EngineId.from_string(engine_id_str)
                except ValueError:
                    log.warning("WorkerPool.sync_from_redis: %s has unknown engine_id=%s",
                                key, engine_id_str)
                    continue
                vram_total_mb = int(payload.get("vram_total_mb", 0))
                health_payload = payload.get("health") or {}
                try:
                    health = EngineHealth(
                        engine_id=eid,
                        state=EngineState(health_payload.get("state", "idle")),
                        vram_used_mb=int(health_payload.get("vram_used_mb", 0)),
                        uptime_seconds=float(health_payload.get("uptime", 0.0)),
                    )
                except (ValueError, KeyError) as exc:
                    log.warning("WorkerPool.sync_from_redis: %s has malformed health (%s)",
                                key, exc)
                    continue
                if worker_id in self.records:
                    self.heartbeat(worker_id, health=health)
                else:
                    self.register(WorkerRecord(
                        worker_id=worker_id,
                        engine_id=eid,
                        vram_total_mb=vram_total_mb,
                        health=health,
                    ))
                updated += 1
            return updated
        except Exception:  # noqa: BLE001 — must not crash the API
            return 0

    def total_capacity_by_engine(self) -> Dict[str, int]:
        out: Dict[str, int] = defaultdict(int)
        with self._lock:
            for r in self.records.values():
                out[r.engine_id.value] += 1
        return dict(out)
