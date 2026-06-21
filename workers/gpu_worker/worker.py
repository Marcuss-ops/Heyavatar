"""GpuWorker dataclass — the worker's main run loop.

Fields and lifecycle methods (``run`` / ``stop`` / ``_update_job_state``)
live here. The per-job processing logic (``_process`` / ``_do_process``)
is attached at import time by :mod:`workers.gpu_worker.process`; this
keeps the run-loop code in this file short and focused on the
reservation / queue-feedback mechanics.

The worker process owns the GPU exclusively. The FastAPI gateway never
imports torch, never allocates VRAM, never blocks the network.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    # Imported only at type-check time so mypy/pyright resolve
    # ``pool: "WorkerPool | None"`` below without paying a runtime
    # import cost in worker processes.
    from src.scheduler.routing.worker_pool import WorkerPool

from contracts.job_queue import JobState, QueueHandle, RenderJob
from src.application.telemetry import TelemetryRecorder
from src.core.config import Settings
from src.domain.enums import EngineId
from src.storage.avatar_packs import AvatarPackRepository
from workers.gpu_worker.telemetry import _bump_inflight, _publish_health


@dataclass(slots=True)
class GpuWorker:
    engine_id: EngineId
    settings: Settings
    pack_repo: AvatarPackRepository
    queue: object  # JobQueue ABC
    handle: QueueHandle
    job_repo: object | None = None  # shared RedisJobRepository for cross-process state
    telemetry: TelemetryRecorder = field(default_factory=TelemetryRecorder)
    # Optional in-process WorkerPool. When set, GpuWorker publishes
    # register/heartbeat/mark_in_flight/unregister around its run loop
    # so the API's TierRouter.pick_available() sees live capacity.
    # Forward ref avoids the circular import between workers.gpu_worker
    # and src.scheduler.routing.worker_pool; ``from __future__ import
    # annotations`` makes this a runtime no-op while mypy still validates.
    pool: WorkerPool | None = None
    # Optional redis client for distributed heartbeat publishing.
    # When set, a daemon thread publishes
    # ``heyavatar:worker:{self.handle.worker_id}:health`` every
    # ``settings.worker_health_publish_seconds`` seconds with a TTL of
    # ``settings.worker_pool_heartbeat_ttl``. The API's
    # ``WorkerPool.sync_from_redis()`` reads the same key-space so
    # TierRouter.pick_available() sees cross-process capacity.
    redis_client: object | None = None
    _stop: bool = field(default=False, init=False)
    _inflight: int = field(default=0, init=False)
    _engine: object | None = field(default=None, init=False, repr=False)
    _hb_thread: object | None = field(default=None, init=False, repr=False)

    def run(self) -> None:
        engine = self._load_engine()
        # Cache the loaded engine on the instance so the single-shot
        # ``publish_heartbeat_once`` debug helper (and any future
        # inspection entry) reuses it instead of re-invoking
        # ``_load_engine`` — which would allocate a second engine
        # instance, leak VRAM, and could disturb a live load.
        self._engine = engine
        repo = self.pack_repo
        queue = self.queue
        # Register ourselves with the WorkerPool (in-process) so capacity
        # tracking is honest. ``self.pool`` is optional — production
        # worker processes that talk via Redis can rely on
        # :meth:`WorkerPool.sync_from_redis` to see us indirectly.
        if self.pool is not None:
            self._register_with_pool(engine)
        # Distributed heartbeat: publish to Redis on a daemon thread so
        # the API process's WorkerPool can see us across process
        # boundaries. Lazy-connects the redis client if not injected.
        self._start_redis_heartbeat(engine)
        try:
            while not self._stop:
                job = queue.reserve(self.handle)
                if job is None:
                    time.sleep(self.settings.job_poll_interval_seconds)
                    continue
                # Cross-process trace context: extract the W3C
                # ``traceparent`` the API stamped on the payload so
                # the worker's span tree continues the API request's
                # trace.
                _bump_inflight(self.engine_id.value, +1)
                if self.pool is not None:
                    self.pool.mark_in_flight(self.handle.worker_id, +1)
                self._update_job_state(job.id, JobState.RUNNING)
                try:
                    final_state, result = self._process(job, engine, repo)
                    queue.acknowledge(job.id)
                    self._update_job_state(job.id, final_state, result=result)
                except Exception as exc:  # pragma: no cover - defensive
                    queue.fail(job.id, reason=f"{type(exc).__name__}: {exc}")
                    self._update_job_state(
                        job.id,
                        JobState.FAILED,
                        result={"error": str(exc)},
                    )
                finally:
                    _bump_inflight(self.engine_id.value, -1)
                    if self.pool is not None:
                        self.pool.mark_in_flight(self.handle.worker_id, -1)
                        # Refresh heart beat so in-flight count + health
                        # snapshot stays consistent for pick_available().
                        try:
                            self.pool.heartbeat(
                                self.handle.worker_id,
                                health=engine.health(),
                            )
                        except Exception:  # pragma: no cover - defensive
                            pass
        finally:
            # Stop the heartbeat thread first so we don't keep writing
            # to Redis after the engine is unloaded.
            self._stop_redis_heartbeat()
            engine.unload()
            if self.pool is not None:
                try:
                    self.pool.unregister(self.handle.worker_id)
                except Exception:  # pragma: no cover - defensive
                    pass
            # Best-effort: drop the worker record from the cluster view
            # so the API's pick_available() stops routing to us
            # within at most ``ttl_seconds``.
            if self.redis_client is not None:
                try:
                    self.redis_client.delete(
                        f"heyavatar:worker:{self.handle.worker_id}:health"
                    )
                except Exception:  # pragma: no cover - defensive
                    pass

    def _register_with_pool(self, engine) -> None:
        """Insert one WorkerRecord for this worker into the in-process pool."""
        try:
            from src.scheduler.routing.worker_pool import WorkerRecord
            self.pool.register(WorkerRecord(
                worker_id=self.handle.worker_id,
                engine_id=self.engine_id,
                vram_total_mb=0,
                health=engine.health(),
            ))
        except Exception as exc:  # pragma: no cover - defensive
            # If the pool is misconfigured we keep running — capacity
            # tracking is best-effort, not a hard dependency.
            from src.core.logging import get_logger
            get_logger(__name__).warning(
                "GpuWorker could not register with WorkerPool: %s", exc,
            )

    # ------------------------------------------------------------------
    # Distributed heartbeat — publish worker health to Redis so the
    # API process's WorkerPool can see us cross-process.
    # ------------------------------------------------------------------
    def _resolve_redis_client(self):
        """Return the injected client, or build one lazily from settings.

        ``redis_client`` is optional on the dataclass. If the operator
        didn't pass one, we connect on demand from ``settings.redis_url``.
        A failure to import ``redis`` or to connect silently degrades
        the worker back to in-process-only capacity tracking.
        """
        if self.redis_client is not None:
            return self.redis_client
        if not self.settings.redis_url:
            return None
        try:
            import redis  # type: ignore
            client = redis.Redis.from_url(
                self.settings.redis_url, decode_responses=True
            )
        except Exception:  # pragma: no cover - defensive
            return None
        return client

    def _start_redis_heartbeat(self, engine) -> None:
        """Spawn a daemon thread that publishes this worker's health.

        The thread polls ``engine.health()`` every
        ``settings.worker_health_publish_seconds`` and writes the
        JSON-snapshot to ``heyavatar:worker:{id}:health`` with a TTL.
        Errors are swallowed inside ``_publish_health`` so a transient
        Redis blip does not crash the worker. Stops cooperatively
        when :attr:`_stop` is set.
        """
        import threading

        client = self._resolve_redis_client()
        if client is None:
            return
        from src.core.logging import get_logger
        log = get_logger("workers.gpu_worker.heartbeat")

        def _loop() -> None:
            period = max(0.1, float(self.settings.worker_health_publish_seconds))
            ttl = int(self.settings.worker_pool_heartbeat_ttl)
            while not self._stop:
                try:
                    health = engine.health()
                except Exception:  # pragma: no cover - engine unloaded mid-tick
                    return
                _publish_health(
                    client,
                    self.handle.worker_id,
                    self.engine_id,
                    health,
                    ttl,
                    log,
                )
                # Sleep in small slices so ``_stop`` short-circuits fast.
                slept = 0.0
                while not self._stop and slept < period:
                    time.sleep(min(0.5, period - slept))
                    slept += 0.5

        thread = threading.Thread(
            target=_loop,
            name=f"worker-heartbeat-{self.handle.worker_id}",
            daemon=True,
        )
        thread.start()
        self._hb_thread = thread

    def _stop_redis_heartbeat(self) -> None:
        """Signal the heartbeat thread to exit and join with a bounded timeout.

        Idempotent: a second call is a no-op. ``self._stop = True`` was
        already set by the reservation loop's ``while not self._stop``
        guard, so the thread sees the exit signal on its next sleep
        slice (≤ 0.5 s).
        """
        thread = self._hb_thread
        if thread is None:
            return
        try:
            thread.join(timeout=2.0)
        except Exception:  # pragma: no cover - defensive
            pass
        self._hb_thread = None

    def publish_heartbeat_once(self, redis_client=None) -> bool:
        """Single-shot publish usable from tests / debug.

        Returns True on a successful SET, False otherwise (no client,
        engine missing, redis SET failed). Does not touch the
        background thread.

        Always uses the engine cached by :meth:`run` so we never
        re-invoke ``_load_engine`` (which would allocate a second
        engine instance, leak VRAM, and could disturb a live load).
        Operators who pre-instantiate a worker without calling
        ``run`` should precede this with an explicit engine load.
        """
        engine = self._engine
        if engine is None:
            # Pre-``run`` call: fall through to the loaded-engine
            # path so test setups that don't drive ``run`` still work,
            # but the production hot-path never hits this branch.
            try:
                engine = self._load_engine()
            except Exception:
                return False
        client = redis_client if redis_client is not None else self._resolve_redis_client()
        if client is None:
            return False
        from src.core.logging import get_logger

        log = get_logger("workers.gpu_worker.heartbeat")
        _publish_health(
            client,
            self.handle.worker_id,
            self.engine_id,
            engine.health(),
            int(self.settings.worker_pool_heartbeat_ttl),
            log,
        )
        return True

    def stop(self) -> None:
        self._stop = True

    def _load_engine(self):
        """Load the engine lazily so the worker boot is not blocked by imports.

        Kept as a method (not a free function) so tests can monkey-patch
        ``GpuWorker._load_engine`` to inject a stub engine.
        """
        from providers import get_provider
        engine = get_provider(self.engine_id)
        engine.load()
        return engine

    def _update_job_state(self, job_id, new_state: JobState, *, result: dict | None = None) -> None:
        if self.job_repo is None:
            return
        try:
            self.job_repo.mark(job_id, new_state, result=result)
        except Exception:
            pass


# ── attach per-job processing methods defined in :mod:`process` ────
# Importing ``process`` at the bottom of this file triggers the
# ``GpuWorker._process = ...`` / ``GpuWorker._do_process = ...``
# assignments there. That way ``from workers.gpu_worker.worker import
# GpuWorker`` always returns a fully-armed class.
from workers.gpu_worker import process  # noqa: E402,F401
