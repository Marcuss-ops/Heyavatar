"""Scenario: all chunks succeed → COMPLETED + output_path."""

from __future__ import annotations

from unittest.mock import patch

from contracts.job_queue import JobState
from providers import get_provider
from src.domain.enums import EngineId
from src.storage.avatar_packs import AvatarPackRepository
from tests._fixtures import PNG_1X1 as _PNG_1x1

from tests.workers.test_gpu_worker._helpers import (
    _build_worker,
    _make_render_job,
    _precompile_identity,
    _successful_render_result,
    requires_ffmpeg,
)


@requires_ffmpeg
def test_do_process_no_degraded_chunks_returns_completed(workdir, tmp_path):
    """No degraded chunks, encoding succeeds → COMPLETED with output_path."""
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

        # Pre-create the manifest file so ``_do_process.is_file()`` passes
        # and the (mocked) ``EncodingWorker.encode()`` actually fires.
        manifest_path = workdir / "captures" / f"{job.id}.manifest.txt"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            f"file '{workdir / 'captures' / (str(job.id) + '_chunk_0000.mp4')}'\n"
        )

        clean_result = _successful_render_result(
            str(job.id), "id-alice", manifest_path=manifest_path
        )
        fake_output = workdir / "captures" / f"{job.id}.mp4"
        fake_output.parent.mkdir(parents=True, exist_ok=True)
        fake_output.write_bytes(b"mock-mp4")

        with patch(
            "src.application.render_video.use_case.RenderVideo.run",
            return_value=clean_result,
        ):
            with patch(
                "workers.encoding_worker.worker.EncodingWorker.encode",
                return_value=fake_output,
            ):
                state, result = worker._do_process(job, engine, pack_repo)

        assert state == JobState.COMPLETED, f"Expected COMPLETED, got {state}"
        assert result["degraded"] is False
        assert result["degraded_chunks"] == []
        assert result["identity_id"] == "id-alice"
        assert result["output_path"] is not None, "output_path should be set"
    finally:
        engine.unload()
