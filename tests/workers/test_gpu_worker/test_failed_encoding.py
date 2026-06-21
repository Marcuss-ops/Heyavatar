"""Scenario: EncodingWorker.encode() raises → FAILED_ENCODING."""

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
def test_do_process_encoding_failure_returns_failed_encoding(workdir, tmp_path):
    """EncodingWorker.encode() raises → FAILED_ENCODING with error."""
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

        clean_chunks = tuple(
            RenderChunkResult(
                chunk_index=0,
                output_path=Path(f"/tmp/chunks/{job.id}_chunk_0000.mp4"),
                duration_seconds=2.0,
                frames_rendered=50,
                gpu_seconds=0.5,
                engine_id=EngineId.MUSE_TALK,
            )
            for _ in range(1)
        )
        # Pre-create the manifest file so ``_do_process.is_file()`` passes
        # and triggers the (mocked) ``EncodingWorker.encode()`` call, which
        # will raise ``Simulated ffmpeg crash`` per the ``side_effect``.
        manifest_path = workdir / "captures" / f"{job.id}.manifest.txt"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            f"file '{workdir / 'captures' / (str(job.id) + '_chunk_0000.mp4')}'\n"
        )

        clean_result = RenderResult(
            job_id=RenderJobId(str(job.id)),
            identity_id=IdentityId("id-alice"),
            output_path=manifest_path,
            duration_seconds=2.0,
            fps=25,
            tier=Tier.EXPRESS,
            engine_id=EngineId.MUSE_TALK,
            gpu_seconds_total=0.5,
            completed_at=datetime.now(timezone.utc),
            chunks=clean_chunks,
            degraded_chunks=(),
        )

        with patch(
            "src.application.render_video.use_case.RenderVideo.run",
            return_value=clean_result,
        ):
            with patch(
                "workers.encoding_worker.worker.EncodingWorker.encode",
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
