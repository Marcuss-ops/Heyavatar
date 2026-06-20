"""GPU worker entrypoint.

A persistent process that loads exactly one :class:`AvatarEngine` instance
and serves a stream of jobs from the configured JobQueue. Each render job
produces chunk videos and a manifest; the :class:`EncodingWorker` is then
invoked to trim overlap, concatenate, mux audio, and produce the final mp4.

The worker process owns the GPU exclusively. The FastAPI gateway never
imports torch, never allocates VRAM, never blocks the network.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from contracts.avatar_engine import EngineState
from contracts.job_queue import JobState, QueueHandle, RenderJob
from src.application import AvatarCompiler, RenderVideo
from src.application.telemetry import TelemetryRecorder
from src.core.config import Settings, get_settings
from src.core.logging import configure_logging, get_logger
from src.domain.enums import EngineId
from src.domain.types import (
    AvatarIdentityHandle,
    IdentitySpec,
    RenderRequest,
    RenderSpec,
)
from src.scheduler.queue import InMemoryJobQueue, NullJobQueue, RedisJobQueue
from src.storage.avatar_packs import AvatarPackRepository
from src.storage.jobs import RedisJobRepository
from providers import get_provider, PROVIDERS
from workers.encoding_worker import EncodingWorker


# ---------------------------------------------------------------------------
# Prometheus exposition — optional. Worker exposes /metrics on
# ``settings.worker_metrics_port`` so Prometheus can scrape it across
# process boundaries. ``prometheus_client`` is a hard requirement at
# runtime — absent it, we degrade silently (the worker still works).
# ---------------------------------------------------------------------------


def _start_metrics_server(port: int) -> None:
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


# ---------------------------------------------------------------------------
# Worker loop
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GpuWorker:
    engine_id: EngineId
    settings: Settings
    pack_repo: AvatarPackRepository
    queue: object  # JobQueue ABC
    handle: QueueHandle
    job_repo: object | None = None  # shared RedisJobRepository for cross-process state
    telemetry: TelemetryRecorder = field(default_factory=TelemetryRecorder)
    _stop: bool = field(default=False, init=False)
    _inflight: int = field(default=0, init=False)

    def run(self) -> None:
        engine = get_provider(self.engine_id)
        engine.load()
        repo = self.pack_repo
        queue = self.queue
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
                self._update_job_state(job.id, JobState.RUNNING)
                try:
                    self._process(job, engine, repo)
                    queue.acknowledge(job.id)
                    self._update_job_state(job.id, JobState.COMPLETED)
                except Exception as exc:  # pragma: no cover - defensive
                    queue.fail(job.id, reason=f"{type(exc).__name__}: {exc}")
                    self._update_job_state(job.id, JobState.FAILED)
                finally:
                    _bump_inflight(self.engine_id.value, -1)
        finally:
            engine.unload()

    def stop(self) -> None:
        self._stop = True

    def _update_job_state(self, job_id, new_state: JobState) -> None:
        if self.job_repo is None:
            return
        try:
            self.job_repo.mark(job_id, new_state)
        except Exception:
            pass

    def _process(self, job: RenderJob, engine, repo: AvatarPackRepository) -> None:
        # Lazy span: only import OTel if present. The tracer is named
        # ``workers.gpu_worker`` so an operator can pivot dashboards
        # by tracer name.
        tracer = None
        try:
            from src.observability.context import extract_traceparent
            from src.observability.tracing import get_tracer
            tracer = get_tracer("workers.gpu_worker")
            parent_ctx = extract_traceparent(job.payload)
        except ImportError:
            parent_ctx = None
            tracer = None

        if tracer is not None:
            from opentelemetry import context as otel_context, trace
            token = otel_context.attach(parent_ctx) if parent_ctx is not None else None
            try:
                with tracer.start_as_current_span(
                    "worker.render_job",
                    kind=trace.SpanKind.CONSUMER,
                    attributes={
                        "heyavatar.engine_id": self.engine_id.value,
                        "heyavatar.worker_id": self.handle.worker_id,
                        "heyavatar.job_id": str(job.id),
                        "heyavatar.tier": str(job.payload.get("tier", "express")),
                    },
                ):
                    self._do_process(job, engine, repo)
            finally:
                if token is not None:
                    otel_context.detach(token)
        else:
            self._do_process(job, engine, repo)

    def _do_process(self, job: RenderJob, engine, repo: AvatarPackRepository) -> None:
        payload = job.payload
        job_type = payload.get("job_type", "render")

        # ── compile-only job ──────────────────────────────────────
        if job_type == "compile":
            source_image = Path(payload["source_image"])
            engine_id_value = payload.get("engine_id", self.engine_id.value)
            spec = IdentitySpec(
                source_image=source_image,
                display_name=payload.get("display_name", ""),
                language_hint=payload.get("language_hint", ""),
            )
            compiler = AvatarCompiler(engine=engine, pack_root=repo.root)
            handle = compiler.compile(spec)
            repo.save(handle.identity_id, read_pack_from_archive(handle.pack_path))
            # Emit a telemetry signal so the operator knows something happened.
            try:
                from src.observability.metrics import record_terminal
                record_terminal(state="compiled", tier="express")
            except ImportError:
                pass
            return

        # ── render job ────────────────────────────────────────────
        identity_id = payload["identity_id"]
        handle = repo.get(_id_from_str(identity_id))
        if handle is None:
            spec = IdentitySpec(
                source_image=Path(payload["source_image"]),
                display_name=payload.get("display_name", ""),
            )
            compiler = AvatarCompiler(engine=engine, pack_root=repo.root)
            handle = compiler.compile(spec)
            repo.save(handle.identity_id, read_pack_from_archive(handle.pack_path))
        request = RenderRequest(
            job_id=job.id,
            identity_id=handle.identity_id,
            identity_spec=IdentitySpec(source_image=Path(payload["source_image"])),
            render_spec=RenderSpec(
                audio_path=Path(payload["audio_path"]),
                fps=int(payload.get("fps", 25)),
            ),
            tier=payload.get("tier", "express"),
        )
        rv = RenderVideo(engine=engine, telemetry=self.telemetry)
        result = rv.run(request, handle)
        # Per-chunk telemetry is published inside RenderVideo.run() —
        # no need to double-publish here. The headline metric
        # ``gpu_seconds_per_minute_of_output`` is computed in Prometheus.
        # ── encoding pass ────────────────────────────────────────
        # The GPU worker renders chunks; the encoding worker trims
        # overlap, concats, and muxes audio into the final mp4.
        manifest_path = result.output_path
        if manifest_path.is_file():
            encoder = EncodingWorker(settings=self.settings)
            try:
                final_path = encoder.encode(
                    str(job.id),
                    manifest_path,
                    audio_path=request.render_spec.audio_path,
                )
                get_logger(__name__).info("encoded final video", extra={"path": str(final_path)})
            except Exception as exc:
                get_logger(__name__).warning("encoding failed; chunks remain in captures", extra={"error": str(exc)})


def _bump_inflight(engine_id: str, delta: int) -> None:
    try:
        from src.observability.metrics import set_inflight
        set_inflight(engine_id, delta)
    except ImportError:
        pass


def _id_from_str(value: str):
    from src.domain.types import IdentityId, RenderJobId
    if value.startswith("job-"):
        return RenderJobId(value)
    return IdentityId(value)


def read_pack_from_archive(path: Path):
    """Convenience wrapper so the worker can re-read a pack it just wrote."""
    from src.domain.avatar_pack import read_pack
    return read_pack(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_queue(settings: Settings):
    if settings.queue_backend == "redis":
        if not settings.redis_url:
            raise RuntimeError("REDIS_URL must be set when queue_backend='redis'.")
        return RedisJobQueue(url=settings.redis_url)
    if settings.queue_backend == "memory":
        return InMemoryJobQueue()
    return NullJobQueue()


def main() -> int:  # pragma: no cover - manual integration
    parser = argparse.ArgumentParser(description="Run a GPU worker for the avatar engine.")
    parser.add_argument("--engine", required=False,
                        help="Engine id to bind to (default: settings.worker_engine_id).")
    parser.add_argument("--worker-id", required=False,
                        help="Worker id used for queue reservation.")
    args = parser.parse_args()

    configure_logging()
    settings = get_settings()
    engine_id = EngineId.from_string(args.engine or settings.worker_engine_id)
    if engine_id not in PROVIDERS:
        print(f"Unknown provider for engine {engine_id}", file=sys.stderr)
        return 2
    # Stand up Prometheus exposition first so we have a heartbeat
    # metric even if the engine fails to load.
    _start_metrics_server(settings.worker_metrics_port)
    try:
        from src.observability.tracing import setup_tracing
        # process_role = worker lets the OTLP exporter label spans
        # per process role on the collector side.
        import os
        os.environ.setdefault("HEYAVATAR_PROCESS_ROLE", "worker")
        setup_tracing(get_settings())
    except ImportError:
        pass

    queue = build_queue(settings)
    handle = QueueHandle(
        worker_id=args.worker_id or settings.worker_id,
        engine_id=engine_id.value,
        tier="any",
    )
    repo = AvatarPackRepository(root=settings.pack_dir)
    job_repo = (
        RedisJobRepository(url=settings.redis_url)
        if settings.queue_backend == "redis" and settings.redis_url
        else None
    )
    worker = GpuWorker(
        engine_id=engine_id,
        settings=settings,
        pack_repo=repo,
        queue=queue,
        handle=handle,
        job_repo=job_repo,
    )
    try:
        worker.run()
    except KeyboardInterrupt:
        worker.stop()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
