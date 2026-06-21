"""Scenario: compile job → COMPLETED with identity_id in result."""

from __future__ import annotations

from datetime import datetime, timezone

from contracts.job_queue import JobState, RenderJob
from src.core.config import get_settings
from src.domain.types import RenderJobId
from src.domain.enums import EngineId
from src.storage.avatar_packs import AvatarPackRepository
from providers import get_provider
from tests._fixtures import PNG_1X1 as _PNG_1x1

from tests.workers.test_gpu_worker._helpers import (
    _build_worker,
    requires_ffmpeg,
)


@requires_ffmpeg
def test_do_process_compiled_job_returns_completed(workdir, tmp_path):
    """Compile job → COMPLETED with identity_id in result."""
    source = tmp_path / "alice.png"
    source.write_bytes(_PNG_1x1)

    pack_repo = AvatarPackRepository(root=workdir / "packs")
    worker = _build_worker(pack_repo=pack_repo)

    engine = get_provider(EngineId.MUSE_TALK)
    engine.load()
    try:
        compile_job = RenderJob(
            id=RenderJobId("job-compile-test"),
            state=JobState.RUNNING,
            payload={
                "job_type": "compile",
                "engine_id": "musetalk-v1",
                "source_image": str(source.resolve()),
                "display_name": "Test",
                "language_hint": "",
            },
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        state, result = worker._do_process(compile_job, engine, pack_repo)
        assert state == JobState.COMPLETED, f"Expected COMPLETED, got {state}"
        assert result["identity_id"].startswith("id-"), "Should contain identity_id"
        assert result["engine_id"] == "musetalk-v1"
        assert "pack_digest" in result
    finally:
        engine.unload()
