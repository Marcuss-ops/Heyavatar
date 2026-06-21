"""Scenario: all chunks degraded → FAILED_INFERENCE."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from contracts.job_queue import JobState
from src.core.config import get_settings
from src.domain.enums import EngineId, Tier
from src.domain.types import (
    IdentityId,
    IdentitySpec,
    RenderChunkResult,
    RenderJobId,
    RenderResult,
)
from src.storage.avatar_packs import AvatarPackRepository
from providers import get_provider
from tests._fixtures import PNG_1X1 as _PNG_1x1

from tests.workers.test_gpu_worker._helpers import (
    _build_worker,
    _make_render_job,
    _precompile_identity,
    requires_ffmpeg,
)


@requires_ffmpeg
def test_do_process_all_chunks_degraded_returns_failed_inference(workdir, tmp_path):
    """All chunks degraded → FAILED_INFERENCE."""
    source = tmp_path / "alice.png"
    source.write_bytes(_PNG_1x1)
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 200 + b"WAVE")

    pack_repo = AvatarPackRepository(root=workdir / "packs")

    engine = get_provider(EngineId.MUSE_TALK)
    engine.load()
    try:
        _precompile_identity(engine, pack_repo, source, identity_id="id-alice")

        worker = _build_worker(pack_repo=pack_repo)
        job = _make_render_job(
            identity_id="id-alice",
            source_image=str(source.resolve()),
            audio_path=str(audio.resolve()),
        )

        # 2 chunks, both degraded → FAILED_INFERENCE
        chunks = tuple(
            RenderChunkResult(
                chunk_index=i,
                output_path=Path(f"/tmp/chunks/{job.id}_degraded_{i:04d}.mp4"),
                duration_seconds=2.0,
                frames_rendered=50,
                gpu_seconds=0.0,
                engine_id=EngineId.MUSE_TALK,
            )
            for i in range(2)
        )
        all_degraded_result = RenderResult(
            job_id=RenderJobId(str(job.id)),
            identity_id=IdentityId("id-alice"),
            output_path=Path(f"./captures/{job.id}.manifest.txt"),
            duration_seconds=4.0,
            fps=25,
            tier=Tier.EXPRESS,
            engine_id=EngineId.MUSE_TALK,
            gpu_seconds_total=0.0,
            completed_at=datetime.now(timezone.utc),
            chunks=chunks,
            degraded_chunks=(0, 1),
        )

        fake_output = workdir / "captures" / f"{job.id}.mp4"
        fake_output.parent.mkdir(parents=True, exist_ok=True)
        fake_output.write_bytes(b"mock-mp4")

        with patch(
            "src.application.render_video.use_case.RenderVideo.run",
            return_value=all_degraded_result,
        ):
            with patch(
                "workers.encoding_worker.worker.EncodingWorker.encode",
                return_value=fake_output,
            ):
                state, result = worker._do_process(job, engine, pack_repo)

        assert state == JobState.FAILED_INFERENCE, (
            f"Expected FAILED_INFERENCE, got {state}"
        )
        assert result["degraded"] is True
        assert result["degraded_chunks"] == [0, 1]
    finally:
        engine.unload()
