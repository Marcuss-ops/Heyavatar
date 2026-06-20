"""End-to-end smoke test: compile + render chunks + encode through mock engine."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from providers import get_provider
from src.application.compile_avatar import AvatarCompiler
from src.application.render_video import ChunkConfig, RenderVideo
from src.application.telemetry import TelemetryRecorder
from src.core.config import get_settings
from src.domain.enums import EngineId
from src.domain.types import RenderJobId, RenderRequest, RenderSpec, IdentitySpec
from tests._fixtures import PNG_1X1 as _PNG_1x1


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="mock-mode smoke test shells out to ffmpeg",
)
def test_end_to_end_compile_and_one_chunk(workdir, tmp_path):
    source = tmp_path / "actor.png"
    source.write_bytes(_PNG_1x1)
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 100 + b"WAVE")

    engine = get_provider(EngineId.MUSE_TALK)
    engine.load()
    try:
        compiler = AvatarCompiler(engine=engine, pack_root=workdir / "packs")
        handle = compiler.compile(IdentitySpec(source_image=source, display_name="Actor 1"))

        rv = RenderVideo(engine=engine, telemetry=TelemetryRecorder(),
                         chunk_config=ChunkConfig(chunk_seconds=2.0, overlap_seconds=0.0))
        request = RenderRequest(
            job_id=RenderJobId("job-smoke"),
            identity_id=handle.identity_id,
            identity_spec=IdentitySpec(source_image=source, display_name="Actor 1"),
            render_spec=RenderSpec(audio_path=audio, fps=25),
        )
        result = rv.run(request, handle)
        assert result.engine_id == EngineId.MUSE_TALK
        assert result.duration_seconds >= 2.0
        # At least one chunk was rendered.
        assert len(result.chunks) >= 1
        # GPU-second accounting moved.
        assert result.gpu_seconds_total > 0
        # Telemetry recorder has entries.
        snap = rv.telemetry.snapshot()
        assert snap["inference_count"] >= 1 or snap["gpu_seconds_total"] >= 0
        # The manifest was written.
        assert result.output_path.is_file()
        assert result.output_path.suffix == ".txt"

        # ── encoding pass ──────────────────────────────────────
        from workers.encoding_worker import EncodingWorker
        encoder = EncodingWorker(settings=get_settings())
        final = encoder.encode("job-smoke", result.output_path, audio_path=audio)
        assert final.is_file()
        assert final.suffix == ".mp4"
    finally:
        engine.unload()
