"""Compile-only flow E2E test.

Verifies the ``POST /avatars/compile`` → job → worker → pack-ready path
without involving the rendering pipeline. Useful to assert that the
compile-only path is independently exercised.
"""

from __future__ import annotations

from datetime import datetime, timezone

from contracts.job_queue import JobState, QueueHandle, RenderJob
from src.application.compile_avatar import AvatarCompiler
from src.domain.avatar_pack import read_pack
from src.domain.types import IdentitySpec, RenderJobId
from src.scheduler.queue.memory import InMemoryJobQueue
from src.storage.avatar_packs import AvatarPackRepository
from src.storage.jobs.memory import InMemoryJobRepository
from providers import get_provider
from src.domain.enums import EngineId
from tests._fixtures import PNG_1X1 as _PNG_1x1


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
