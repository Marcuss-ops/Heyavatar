"""GPU worker entrypoint.

A persistent process that loads exactly one :class:`AvatarEngine` instance
and serves a stream of jobs from the configured JobQueue. Each job is
processed by :class:`RenderVideo` and ack'd on success or failed on
exception.

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
from contracts.job_queue import JobState, QueueHandle
from src.application import AvatarCompiler, RenderVideo
from src.application.telemetry import TelemetryRecorder
from src.core.config import Settings, get_settings
from src.core.logging import configure_logging
from src.domain.enums import EngineId
from src.domain.types import (
    AvatarIdentityHandle,
    IdentitySpec,
    RenderRequest,
    RenderSpec,
)
from src.scheduler.queue import InMemoryJobQueue, NullJobQueue, RedisJobQueue
from src.storage.avatar_packs import AvatarPackRepository
from providers import get_provider, PROVIDERS


@dataclass(slots=True)
class GpuWorker:
    engine_id: EngineId
    settings: Settings
    pack_repo: AvatarPackRepository
    queue: object  # JobQueue ABC
    handle: QueueHandle
    _stop: bool = field(default=False, init=False)

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
                try:
                    self._process(job.payload, engine, repo)
                    queue.acknowledge(job.id)
                except Exception as exc:  # pragma: no cover - defensive
                    queue.fail(job.id, reason=f"{type(exc).__name__}: {exc}")
        finally:
            engine.unload()

    def stop(self) -> None:
        self._stop = True

    def _process(self, payload: dict, engine, repo: AvatarPackRepository) -> None:
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
            job_id=_id_from_str(payload["job_id"]),
            identity_id=handle.identity_id,
            identity_spec=IdentitySpec(source_image=Path(payload["source_image"])),
            render_spec=RenderSpec(
                audio_path=Path(payload["audio_path"]),
                fps=int(payload.get("fps", 25)),
            ),
            tier=payload.get("tier", "express"),
        )
        rv = RenderVideo(engine=engine, telemetry=TelemetryRecorder())
        result = rv.run(request, handle)
        # The encoder worker would now take ``result.output_path`` and produce
        # the final mp4. In this single-process demo we just keep the file.


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
    queue = build_queue(settings)
    handle = QueueHandle(
        worker_id=args.worker_id or settings.worker_id,
        engine_id=engine_id.value,
        tier="any",
    )
    repo = AvatarPackRepository(root=settings.pack_dir)
    worker = GpuWorker(
        engine_id=engine_id,
        settings=settings,
        pack_repo=repo,
        queue=queue,
        handle=handle,
    )
    try:
        worker.run()
    except KeyboardInterrupt:
        worker.stop()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
