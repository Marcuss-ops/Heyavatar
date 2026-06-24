"""Shared helpers for :class:`GpuWorker` state-transition tests.

* :data:`requires_ffmpeg` — pytest marker that skips without ffmpeg.
* :func:`_make_render_job` — fabricate a minimal :class:`RenderJob`.
* :func:`_build_worker` — assemble a :class:`GpuWorker` for testing.
* :func:`_precompile_identity` — helper that runs through the compile
  flow so subsequent render jobs can resolve a pack.
* :func:`_make_chunk` / :func:`_successful_render_result` — fixture
  factories for happy / degraded :class:`RenderResult` objects.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pytest

from contracts.job_queue import JobState, QueueHandle, RenderJob
from src.core.config import get_settings
from src.domain.enums import EngineId, Tier
from src.domain.types import (
    IdentityId,
    IdentitySpec,
    RenderChunkResult,
    RenderJobId,
    RenderResult,
)
from src.scheduler.queue.memory import InMemoryJobQueue
from src.storage.avatar_packs import AvatarPackRepository
from workers.gpu_worker.worker import GpuWorker


requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="test shells out to ffmpeg for mock render",
)


def _make_render_job(
    *,
    job_id: str = "job-worker-state-001",
    identity_id: str = "id-alice",
    source_image: str = "",
    audio_path: str = "",
    motion_style: str | None = None,
) -> RenderJob:
    """Build a minimal render job for _do_process."""
    now = datetime.now(timezone.utc)
    return RenderJob(
        id=RenderJobId(job_id),
        state=JobState.RUNNING,
        payload={
            "job_type": "render",
            "identity_id": identity_id,
            "source_image": source_image,
            "audio_path": audio_path,
            "fps": 25,
            "tier": "express",
            "motion_style": motion_style,
        },
        created_at=now,
        updated_at=now,
    )


def _build_worker(*, pack_repo: AvatarPackRepository) -> GpuWorker:
    """Build a minimal GpuWorker for testing."""
    settings = get_settings()
    return GpuWorker(
        engine_id=EngineId.MUSE_TALK,
        settings=settings,
        pack_repo=pack_repo,
        queue=InMemoryJobQueue(),
        handle=QueueHandle(worker_id="w-test", engine_id="musetalk-v1", tier="any"),
        job_repo=None,
    )


def _precompile_identity(
    engine,
    pack_repo: AvatarPackRepository,
    source: Path,
    identity_id: str = "id-alice",
) -> None:
    """Compile an identity and persist to the pack repo under the given id."""
    from src.application.compile_avatar import AvatarCompiler
    from src.domain.avatar_pack import read_pack

    spec = IdentitySpec(source_image=source, display_name="Test")
    compiler = AvatarCompiler(engine=engine, pack_root=pack_repo.root)
    handle = compiler.compile(spec)
    pack_repo.save(IdentityId(identity_id), read_pack(handle.pack_path))


def _make_chunk(index: int) -> RenderChunkResult:
    return RenderChunkResult(
        chunk_index=index,
        output_path=Path(f"/tmp/chunks/chunk_{index:04d}.mp4"),
        duration_seconds=2.0,
        frames_rendered=50,
        gpu_seconds=0.5,
        engine_id=EngineId.MUSE_TALK,
    )


def _successful_render_result(
    job_id: str,
    identity_id: str = "id-alice",
    *,
    num_chunks: int = 1,
    degraded_chunks: tuple[int, ...] = (),
    manifest_path: Optional[Path] = None,
    duration_seconds: float = 2.0,
) -> RenderResult:
    """Build a fully-populated all-success :class:`RenderResult` for tests.

    Mirrors the schema :meth:`RenderVideo.run` returns on the happy path.
    Use ``degraded_chunks`` for the COMPLETED_DEGRADED scenario.
    """
    chunks = tuple(
        RenderChunkResult(
            chunk_index=i,
            output_path=Path(f"/tmp/chunks/{job_id}_chunk_{i:04d}.mp4"),
            duration_seconds=duration_seconds,
            frames_rendered=int(duration_seconds * 25),
            gpu_seconds=0.5,
            engine_id=EngineId.MUSE_TALK,
        )
        for i in range(num_chunks)
    )
    return RenderResult(
        job_id=RenderJobId(job_id),
        identity_id=IdentityId(identity_id),
        output_path=manifest_path or Path(f"./captures/{job_id}.manifest.txt"),
        duration_seconds=duration_seconds * num_chunks,
        fps=25,
        tier=Tier.EXPRESS,
        engine_id=EngineId.MUSE_TALK,
        gpu_seconds_total=0.5 * num_chunks,
        completed_at=datetime.now(timezone.utc),
        chunks=chunks,
        degraded_chunks=degraded_chunks,
    )
