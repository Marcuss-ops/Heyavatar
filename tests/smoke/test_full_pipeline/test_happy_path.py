"""Happy-path E2E test: full API → Worker → Encoder → Status flow.

Walks a render job through every stage of the pipeline (compile, render,
encode, acknowledge, status) using the exact same code paths as
production, just with ``InMemoryJobQueue`` / ``InMemoryJobRepository``
instead of Redis. This is the test that proves Blocco 1 is complete.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from contracts.job_queue import JobState, QueueHandle, RenderJob
from src.application.compile_avatar import AvatarCompiler
from src.application.render_video.config import ChunkConfig
from src.application.render_video.use_case import RenderVideo
from src.application.telemetry import TelemetryRecorder
from src.core.config import get_settings
from src.domain.avatar_pack import read_pack
from src.domain.enums import EngineId, Tier
from src.domain.types import (
    IdentitySpec,
    RenderJobId,
    RenderRequest,
    RenderSpec,
)
from src.scheduler.queue.memory import InMemoryJobQueue
from src.storage.avatar_packs import AvatarPackRepository
from src.storage.jobs.memory import InMemoryJobRepository
from providers import get_provider
from workers.encoding_worker.worker import EncodingWorker
from workers.gpu_worker.telemetry import _id_from_str
from tests._fixtures import PNG_1X1 as _PNG_1x1

from tests.smoke.test_full_pipeline._helpers import (
    requires_ffmpeg,
    _publish_render_job,
    _simulate_worker_reserve,
)


@requires_ffmpeg
def test_full_pipeline_api_to_worker_to_encoder_to_status(workdir, tmp_path):
    """Walk a render job through the full single-process pipeline."""
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
