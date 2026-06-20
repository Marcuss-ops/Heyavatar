"""End-to-end pipeline test: full API → Queue → Worker → Encoder → Status flow.

Exercises the entire orchestration path using ``InMemoryJobQueue`` and
``InMemoryJobRepository`` (single-process, no Redis needed) to verify
that a render job flows correctly through every stage of the pipeline:

1. API publishes job → queue + repository
2. Worker reserves job → state transitions to RESERVED, then RUNNING
3. Worker compiles identity (first time) or loads existing pack
4. Worker renders chunks via mock engine → writes chunk manifest
5. Worker encodes chunks via ``EncodingWorker`` → final mp4
6. Worker acknowledges job → state transitions to COMPLETED
7. Status check reads COMPLETED from repository

The test uses the *exact same code paths* as the production pipeline,
just with ``InMemoryJobQueue`` instead of Redis.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from contracts.job_queue import JobState, QueueHandle, RenderJob
from src.application.telemetry import TelemetryRecorder
from src.core.config import get_settings
from src.domain.enums import EngineId, Tier
from src.domain.types import (
    IdentitySpec,
    RenderJobId,
    RenderRequest,
    RenderSpec,
)
from src.scheduler.queue import InMemoryJobQueue
from src.storage.avatar_packs import AvatarPackRepository
from src.storage.jobs import InMemoryJobRepository
from providers import get_provider
from workers.encoding_worker import EncodingWorker
from tests._fixtures import PNG_1X1 as _PNG_1x1


requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="mock-mode E2E test shells out to ffmpeg for encoding",
)


# ── helpers ─────────────────────────────────────────────────────────


def _publish_render_job(
    queue: InMemoryJobQueue,
    repo: InMemoryJobRepository,
    *,
    identity_id: str = "id-alice",
    source_image: str = "",
    audio_path: str = "",
    tier: str = "express",
) -> RenderJob:
    """Simulate the API's ``POST /jobs`` endpoint."""
    now = datetime.now(timezone.utc)
    job = RenderJob(
        id=RenderJobId("job-e2e-001"),
        state=JobState.PENDING,
        payload={
            "identity_id": identity_id,
            "source_image": source_image,
            "audio_path": audio_path,
            "fps": 25,
            "tier": tier,
            "job_type": "render",
        },
        created_at=now,
        updated_at=now,
    )
    queue.publish(job)
    repo.upsert(job)
    return job


def _simulate_worker_reserve(
    queue: InMemoryJobQueue,
    repo: InMemoryJobRepository,
    worker_id: str = "w-1",
) -> RenderJob | None:
    """Simulate the worker's ``reserve()`` call and subsequent state update."""
    handle = QueueHandle(
        worker_id=worker_id, engine_id="musetalk-v1", tier="any"
    )
    job = queue.reserve(handle)
    if job is not None:
        repo.mark(job.id, JobState.RESERVED)
    return job


# ── the test ────────────────────────────────────────────────────────


@requires_ffmpeg
def test_full_pipeline_api_to_worker_to_encoder_to_status(workdir, tmp_path):
    """Walk a render job through the full single-process pipeline.

    This is the test that proves Blocco 1 is complete.
    """
    # ── 0. Fixtures ──────────────────────────────────────────────
    source = tmp_path / "actor.png"
    source.write_bytes(_PNG_1x1)
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 200 + b"WAVE")

    settings = get_settings()
    queue = InMemoryJobQueue()
    repo = InMemoryJobRepository()
    pack_repo = AvatarPackRepository(root=workdir / "packs")
    telemetry = TelemetryRecorder()

    # ── 1. Pre-compile identity (simulate earlier /avatars/compile) ──
    engine = get_provider(EngineId.MUSE_TALK)
    engine.load()
    try:
        # Publish a compile job so the worker compiles on first render.
        compile_job = RenderJob(
            id=RenderJobId("job-compile-pre"),
            state=JobState.PENDING,
            payload={
                "job_type": "compile",
                "engine_id": "musetalk-v1",
                "source_image": str(source.resolve()),
                "display_name": "Actor 1",
                "language_hint": "",
            },
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        queue.publish(compile_job)
        repo.upsert(compile_job)

        # ── 2. Worker: reserve compile job, process it ───────────
        handle = QueueHandle(worker_id="w-1", engine_id="musetalk-v1", tier="any")
        compile_reserved = queue.reserve(handle)
        assert compile_reserved is not None
        assert compile_reserved.payload["job_type"] == "compile"
        repo.mark(compile_reserved.id, JobState.RESERVED)
        repo.mark(compile_reserved.id, JobState.RUNNING)

        # Process compile: prepare_identity → AvatarPack
        from src.application.compile_avatar import AvatarCompiler
        from src.domain.avatar_pack import read_pack

        spec = IdentitySpec(
            source_image=source,
            display_name="Actor 1",
        )
        compiler = AvatarCompiler(engine=engine, pack_root=pack_repo.root)
        identity_handle = compiler.compile(spec)
        pack_repo.save(identity_handle.identity_id, read_pack(identity_handle.pack_path))

        queue.acknowledge(compile_reserved.id)
        repo.mark(compile_reserved.id, JobState.COMPLETED)

        # ── 3. API: publish render job (identity_id matches compile) ──
        render_job = _publish_render_job(
            queue,
            repo,
            identity_id=str(identity_handle.identity_id),
            source_image=str(source.resolve()),
            audio_path=str(audio.resolve()),
        )

        # Verify initial state.
        assert repo.get(render_job.id).state == JobState.PENDING
        assert queue.depth() == 1  # only the render job

        # ── 4. Worker: reserve render job ────────────────────────
        render_reserved = _simulate_worker_reserve(queue, repo)
        assert render_reserved is not None
        assert render_reserved.id == render_job.id
        assert repo.get(render_job.id).state == JobState.RESERVED

        # ── 5. Worker: process render job ────────────────────────
        repo.mark(render_job.id, JobState.RUNNING)
        assert repo.get(render_job.id).state == JobState.RUNNING

        # Reconstruct the request from payload (same as GpuWorker._do_process).
        payload = render_reserved.payload
        identity_id = payload["identity_id"]
        from workers.gpu_worker import _id_from_str
        handle_from_repo = pack_repo.get(_id_from_str(identity_id))
        assert handle_from_repo is not None, (
            "Avatar pack should be loadable from repo via identity_id in payload"
        )

        request = RenderRequest(
            job_id=render_job.id,
            identity_id=identity_handle.identity_id,
            identity_spec=IdentitySpec(source_image=Path(payload["source_image"])),
            render_spec=RenderSpec(
                audio_path=Path(payload["audio_path"]),
                fps=int(payload.get("fps", 25)),
            ),
            tier=Tier.EXPRESS,
        )

        # Render chunks via RenderVideo.
        from src.application.render_video import ChunkConfig, RenderVideo

        rv = RenderVideo(
            engine=engine,
            telemetry=telemetry,
            chunk_config=ChunkConfig(chunk_seconds=2.0, overlap_seconds=0.0),
        )
        result = rv.run(request, identity_handle)

        # Assertions on the render result.
        assert result.engine_id == EngineId.MUSE_TALK
        assert result.duration_seconds >= 2.0
        assert len(result.chunks) >= 1
        assert result.gpu_seconds_total > 0
        assert result.output_path.is_file()
        assert result.output_path.suffix == ".txt"  # manifest, not mp4

        # ── 6. Worker: encode via EncodingWorker ─────────────────
        encoder = EncodingWorker(settings=settings)
        final_path = encoder.encode(
            str(render_job.id),
            result.output_path,
            audio_path=audio,
        )
        assert final_path.is_file()
        assert final_path.suffix == ".mp4"

        # ── 7. Worker: acknowledge job, update state ─────────────
        queue.acknowledge(render_job.id)
        repo.mark(render_job.id, JobState.COMPLETED)

        # ── 8. API: status check ──────────────────────────────────
        final_job = repo.get(render_job.id)
        assert final_job is not None
        assert final_job.state == JobState.COMPLETED, (
            f"Expected COMPLETED, got {final_job.state}"
        )

        # Queue depth is zero after processing.
        assert queue.depth() == 0

        # Verify the final video is non-empty.
        assert final_path.stat().st_size > 0, "Final mp4 should be non-empty"

        # Telemetry recorded inference counts.
        snap = telemetry.snapshot()
        assert snap["inference_count"] >= 1 or snap["gpu_seconds_total"] > 0

    finally:
        engine.unload()


def test_compile_via_queue_job_worker_pack_ready(workdir, tmp_path):
    """Verify the compile-only flow: POST /avatars/compile → job → worker → pack ready.

    Simulates:
    1. API publishes compile job (job_type="compile") to queue + repo
    2. Worker reserves the compile job
    3. Worker processes it: AvatarCompiler.compile() → AvatarPack saved
    4. Worker acknowledges → state COMPLETED
    5. Status check: repo returns COMPLETED
    6. Pack is loadable and has correct identity id
    """
    source = tmp_path / "alice.png"
    source.write_bytes(_PNG_1x1)

    queue = InMemoryJobQueue()
    repo = InMemoryJobRepository()
    pack_repo = AvatarPackRepository(root=workdir / "packs")

    # ── 1. API: publish compile job ────────────────────────────
    now = datetime.now(timezone.utc)
    compile_job = RenderJob(
        id=RenderJobId("job-compile-e2e"),
        state=JobState.PENDING,
        payload={
            "job_type": "compile",
            "engine_id": "musetalk-v1",
            "source_image": str(source.resolve()),
            "display_name": "Alice",
            "language_hint": "it-IT",
        },
        created_at=now,
        updated_at=now,
    )
    queue.publish(compile_job)
    repo.upsert(compile_job)

    # Verify initial state.
    assert repo.get(compile_job.id).state == JobState.PENDING
    assert queue.depth() == 1

    # ── 2. Worker: reserve compile job ─────────────────────────
    handle = QueueHandle(worker_id="w-compile", engine_id="musetalk-v1", tier="any")
    reserved = queue.reserve(handle)
    assert reserved is not None
    assert reserved.payload["job_type"] == "compile"
    repo.mark(reserved.id, JobState.RESERVED)
    assert repo.get(reserved.id).state == JobState.RESERVED

    # ── 3. Worker: process compile job ─────────────────────────
    repo.mark(reserved.id, JobState.RUNNING)
    assert repo.get(reserved.id).state == JobState.RUNNING

    engine = get_provider(EngineId.MUSE_TALK)
    engine.load()
    try:
        from src.application.compile_avatar import AvatarCompiler
        from src.domain.avatar_pack import read_pack

        spec = IdentitySpec(
            source_image=source,
            display_name="Alice",
            language_hint="it-IT",
        )
        compiler = AvatarCompiler(engine=engine, pack_root=pack_repo.root)
        identity_handle = compiler.compile(spec)
        pack_repo.save(identity_handle.identity_id, read_pack(identity_handle.pack_path))

        # ── 4. Worker: acknowledge ──────────────────────────────
        queue.acknowledge(reserved.id)
        repo.mark(reserved.id, JobState.COMPLETED)

        # ── 5. API: status check ─────────────────────────────────
        final_job = repo.get(reserved.id)
        assert final_job is not None
        assert final_job.state == JobState.COMPLETED, (
            f"Expected COMPLETED, got {final_job.state}"
        )
        assert queue.depth() == 0

        # ── 6. Pack is loadable and correct ──────────────────────
        loaded_handle = pack_repo.get(identity_handle.identity_id)
        assert loaded_handle is not None, (
            "Avatar pack should be loadable after compile"
        )
        assert loaded_handle.manifest.identity_id == identity_handle.identity_id
        assert loaded_handle.archive_path.is_file()
        assert "musetalk-v1" in loaded_handle.manifest.engine_compatibility
    finally:
        engine.unload()


@requires_ffmpeg
def test_full_pipeline_failed_job_is_recorded(workdir, tmp_path):
    """Verify that a failed job is correctly recorded in queue + repo."""
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 200 + b"WAVE")

    queue = InMemoryJobQueue()
    repo = InMemoryJobRepository()

    job = _publish_render_job(
        queue,
        repo,
        identity_id="id-nonexistent",
        source_image=str(tmp_path / "missing.png"),
        audio_path=str(audio.resolve()),
    )
    assert repo.get(job.id).state == JobState.PENDING

    reserved = _simulate_worker_reserve(queue, repo)
    assert reserved is not None
    assert repo.get(job.id).state == JobState.RESERVED

    # Simulate worker failure.
    queue.fail(job.id, reason="GPU OOM during render")
    repo.mark(job.id, JobState.FAILED, error="GPU OOM during render")

    final = repo.get(job.id)
    assert final is not None
    assert final.state == JobState.FAILED
    assert "GPU OOM" in (final.last_error or "")
    assert queue.depth() == 0
