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

    def total_capacity_by_engine(self) -> Dict[str, int]:
        out: Dict[str, int] = defaultdict(int)
        with self._lock:
            for r in self.records.values():
                out[r.engine_id.value] += 1
        return dict(out)
