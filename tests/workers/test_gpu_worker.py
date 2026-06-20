"""Worker state transition tests for :meth:`GpuWorker._do_process`.

Verifies that the worker returns the correct ``JobState`` for each
terminal outcome: COMPLETED, COMPLETED_DEGRADED, FAILED_INFERENCE,
and FAILED_ENCODING.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from contracts.job_queue import JobState, QueueHandle, RenderJob
from src.application.render_video import RenderResult
from src.core.config import get_settings
from src.domain.enums import EngineId, Tier
from src.domain.types import (
    IdentityId,
    IdentitySpec,
    RenderChunkResult,
    RenderJobId,
)
from src.scheduler.queue import InMemoryJobQueue
from src.storage.avatar_packs import AvatarPackRepository
from workers.gpu_worker import GpuWorker
from tests._fixtures import PNG_1X1 as _PNG_1x1

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="test shells out to ffmpeg for mock render",
)


# ── helpers ─────────────────────────────────────────────────────────


def _make_render_job(
    *,
    job_id: str = "job-worker-state-001",
    identity_id: str = "id-alice",
    source_image: str = "",
    audio_path: str = "",
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
    identity_id: str,
    *,
    num_chunks: int = 1,
    degraded_chunks: tuple[int, ...] = (),
    manifest_path: Path | None = None,
) -> RenderResult:
    """Build a RenderResult with controlled chunk/degraded counts."""
    chunks = tuple(_make_chunk(i) for i in range(num_chunks))
    mp = manifest_path or Path(f"./captures/{job_id}.manifest.txt")
    return RenderResult(
        job_id=RenderJobId(job_id),
        identity_id=IdentityId(identity_id),
        output_path=mp,
        duration_seconds=2.0 * num_chunks,
        fps=25,
        tier=Tier.EXPRESS,
        engine_id=EngineId.MUSE_TALK,
        gpu_seconds_total=0.5 * num_chunks,
        completed_at=datetime.now(timezone.utc),
        chunks=chunks,
        degraded_chunks=degraded_chunks,
    )


# ── tests ───────────────────────────────────────────────────────────


@requires_ffmpeg
def test_do_process_compiled_job_returns_completed(workdir, tmp_path):
    """Compile job → COMPLETED with identity_id in result."""
    source = tmp_path / "alice.png"
    source.write_bytes(_PNG_1x1)

    pack_repo = AvatarPackRepository(root=workdir / "packs")
    worker = _build_worker(pack_repo=pack_repo)

    from providers import get_provider
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


@requires_ffmpeg
def test_do_process_no_degraded_chunks_returns_completed(workdir, tmp_path):
    """No degraded chunks, encoding succeeds → COMPLETED with output_path."""
    source = tmp_path / "alice.png"
    source.write_bytes(_PNG_1x1)
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 200 + b"WAVE")

    pack_repo = AvatarPackRepository(root=workdir / "packs")

    from providers import get_provider
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

        clean_result = _successful_render_result(str(job.id), "id-alice")
        fake_output = workdir / "captures" / f"{job.id}.mp4"
        fake_output.parent.mkdir(parents=True, exist_ok=True)
        fake_output.write_bytes(b"mock-mp4")

        with patch(
            "src.application.render_video.RenderVideo.run",
            return_value=clean_result,
        ):
            with patch(
                "workers.encoding_worker.EncodingWorker.encode",
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


@requires_ffmpeg
def test_do_process_degraded_chunks_returns_completed_degraded(workdir, tmp_path):
    """One chunk degraded → COMPLETED_DEGRADED with degraded=True."""
    source = tmp_path / "alice.png"
    source.write_bytes(_PNG_1x1)
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 200 + b"WAVE")

    pack_repo = AvatarPackRepository(root=workdir / "packs")

    from providers import get_provider
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

        degraded_result = _successful_render_result(
            str(job.id), "id-alice", num_chunks=2, degraded_chunks=(1,)
        )
        fake_output = workdir / "captures" / f"{job.id}.mp4"
        fake_output.parent.mkdir(parents=True, exist_ok=True)
        fake_output.write_bytes(b"mock-mp4")

        with patch(
            "src.application.render_video.RenderVideo.run",
            return_value=degraded_result,
        ):
            with patch(
                "workers.encoding_worker.EncodingWorker.encode",
                return_value=fake_output,
            ):
                state, result = worker._do_process(job, engine, pack_repo)

        assert state == JobState.COMPLETED_DEGRADED, (
            f"Expected COMPLETED_DEGRADED, got {state}"
        )
        assert result["degraded"] is True
        assert result["degraded_chunks"] == [1]
        assert result["total_chunks"] == 2
        assert result["identity_id"] == "id-alice"
    finally:
        engine.unload()


@requires_ffmpeg
def test_do_process_all_chunks_degraded_returns_failed_inference(workdir, tmp_path):
    """All chunks degraded → FAILED_INFERENCE."""
    source = tmp_path / "alice.png"
    source.write_bytes(_PNG_1x1)
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 200 + b"WAVE")

    pack_repo = AvatarPackRepository(root=workdir / "packs")

    from providers import get_provider
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
        degraded_result = _successful_render_result(
            str(job.id), "id-alice", num_chunks=2, degraded_chunks=(0, 1)
        )
        fake_output = workdir / "captures" / f"{job.id}.mp4"
        fake_output.parent.mkdir(parents=True, exist_ok=True)
        fake_output.write_bytes(b"mock-mp4")

        with patch(
            "src.application.render_video.RenderVideo.run",
            return_value=degraded_result,
        ):
            with patch(
                "workers.encoding_worker.EncodingWorker.encode",
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


@requires_ffmpeg
def test_do_process_encoding_failure_returns_failed_encoding(workdir, tmp_path):
    """EncodingWorker.encode() raises → FAILED_ENCODING with error."""
    source = tmp_path / "alice.png"
    source.write_bytes(_PNG_1x1)
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 200 + b"WAVE")

    pack_repo = AvatarPackRepository(root=workdir / "packs")

    from providers import get_provider
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

        clean_result = _successful_render_result(str(job.id), "id-alice")

        with patch(
            "src.application.render_video.RenderVideo.run",
            return_value=clean_result,
        ):
            with patch(
                "workers.encoding_worker.EncodingWorker.encode",
                side_effect=RuntimeError("Simulated ffmpeg crash"),
            ):
                state, result = worker._do_process(job, engine, pack_repo)

        assert state == JobState.FAILED_ENCODING, (
            f"Expected FAILED_ENCODING, got {state}"
        )
        assert "Simulated ffmpeg crash" in result.get("error", "")
        assert result["identity_id"] == "id-alice"
        assert result["degraded"] is False
    finally:
        engine.unload()
