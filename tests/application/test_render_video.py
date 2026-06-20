"""Chunk retry tests for :class:`RenderVideo`.

Verifies that a failed chunk is retried up to ``chunk_retry_max`` times,
that exhausted retries produce a degraded fallback, and that the job
completes with a mix of successful and degraded chunks.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from contracts.avatar_engine import AvatarEngine, EngineState
from src.application.render_video import ChunkConfig, RenderVideo
from src.application.telemetry import TelemetryRecorder
from src.core.config import get_settings
from src.domain.enums import EngineId, Tier
from src.domain.types import (
    AvatarIdentityHandle,
    IdentityId,
    IdentitySpec,
    RenderChunkRequest,
    RenderChunkResult,
    RenderJobId,
    RenderRequest,
    RenderSpec,
)
from tests._fixtures import PNG_1X1 as _PNG_1x1


requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="test shells out to ffmpeg for degraded mp4 stub",
)


# ── helpers ─────────────────────────────────────────────────────────


def _mock_identity() -> AvatarIdentityHandle:
    """Return a minimal identity handle for test requests."""
    return AvatarIdentityHandle(
        identity_id=IdentityId("id-test"),
        pack_path=Path("/tmp/mock.pack"),
        pack_digest="sha256:deadbeef",
        prepared_at=datetime.now(timezone.utc),
    )


def _make_request(
    *,
    audio_path: Path | None = None,
    chunk_seconds: float = 2.0,
) -> RenderRequest:
    """Build a minimal RenderRequest with a controlled audio path."""
    audio = audio_path or Path("/tmp/test.wav")
    return RenderRequest(
        job_id=RenderJobId("job-retry-001"),
        identity_id=IdentityId("id-test"),
        identity_spec=IdentitySpec(source_image=Path("/tmp/source.png")),
        render_spec=RenderSpec(audio_path=audio, fps=25, target_resolution=(512, 512)),
        tier=Tier.EXPRESS,
    )


def _make_chunk_config(
    *,
    retry_max: int = 3,
    retry_delay: float = 0.0,
    chunk_seconds: float = 2.0,
    overlap_seconds: float = 0.0,
) -> ChunkConfig:
    return ChunkConfig(
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
        chunk_retry_max=retry_max,
        chunk_retry_delay_seconds=retry_delay,
    )


def _successful_chunk(index: int, job_id: str = "job-retry-001") -> RenderChunkResult:
    """A valid render result for a single chunk."""
    return RenderChunkResult(
        chunk_index=index,
        output_path=Path(f"/tmp/chunks/{job_id}_chunk_{index:04d}.mp4"),
        duration_seconds=2.0,
        frames_rendered=50,
        gpu_seconds=0.5,
        engine_id=EngineId.MUSE_TALK,
    )


class FailingEngine(AvatarEngine):
    """Mock engine that fails exactly ``fail_count`` times, then succeeds.

    When ``fail_count`` reaches zero, subsequent calls return
    ``_successful_chunk()``. If ``always_fail=True`` the counter is
    ignored and every call raises.
    """

    engine_id = EngineId.MUSE_TALK

    def __init__(self, *, fail_count: int = 0, always_fail: bool = False):
        super().__init__()
        self.fail_count = fail_count
        self.always_fail = always_fail
        self.settings = get_settings()
        self._state = EngineState.IDLE

    def load(self) -> None:
        self.mark_loaded()

    def unload(self) -> None:
        pass

    def prepare_identity(self, source_image: Path) -> AvatarIdentityHandle:
        return _mock_identity()

    def render_chunk(
        self,
        request: RenderChunkRequest,
        identity: AvatarIdentityHandle,
    ) -> RenderChunkResult:
        if self.always_fail or self.fail_count > 0:
            if self.fail_count > 0:
                self.fail_count -= 1
            raise RuntimeError(
                f"Simulated GPU OOM on chunk {request.chunk_index}"
            )
        return _successful_chunk(request.chunk_index)


class SelectiveFailingEngine(AvatarEngine):
    """Engine that fails ONLY for a specific chunk index."""

    engine_id = EngineId.MUSE_TALK

    def __init__(self, *, failing_index: int):
        super().__init__()
        self.failing_index = failing_index
        self.settings = get_settings()
        self._state = EngineState.IDLE

    def load(self) -> None:
        self.mark_loaded()

    def unload(self) -> None:
        pass

    def prepare_identity(self, source_image: Path) -> AvatarIdentityHandle:
        return _mock_identity()

    def render_chunk(
        self,
        request: RenderChunkRequest,
        identity: AvatarIdentityHandle,
    ) -> RenderChunkResult:
        if request.chunk_index == self.failing_index:
            raise RuntimeError("Simulated crash on chunk 1")
        return _successful_chunk(request.chunk_index)


# ── tests ───────────────────────────────────────────────────────────


@requires_ffmpeg
def test_retry_succeeds_on_last_attempt(workdir, tmp_path):
    """Engine fails 2 times, succeeds on 3rd — chunk rendered, no degraded."""
    source = tmp_path / "src.png"
    source.write_bytes(_PNG_1x1)
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 200 + b"WAVE")

    engine = FailingEngine(fail_count=2)  # fail twice, then succeed
    engine.load()
    telemetry = TelemetryRecorder()
    cfg = _make_chunk_config(retry_max=3, retry_delay=0.0, chunk_seconds=2.0)
    rv = RenderVideo(engine=engine, telemetry=telemetry, chunk_config=cfg)
    identity = _mock_identity()
    request = _make_request(audio_path=audio, chunk_seconds=2.0)

    # Control chunk count: mock _probe_audio_duration to return exactly
    # one chunk's worth of audio.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "src.application.render_video._probe_audio_duration",
            lambda _: 2.0,
        )
        result = rv.run(request, identity)

    # Single chunk, rendered successfully after 2 retries.
    assert len(result.chunks) == 1
    chunk = result.chunks[0]
    assert "degraded" not in chunk.output_path.name
    assert chunk.gpu_seconds > 0
    assert engine.fail_count == 0  # exhausted its failures

    # Telemetry recorded the successful render.
    assert telemetry.gpu_seconds_total > 0, (
        "Telemetry should record GPU-seconds for successful retry"
    )


@requires_ffmpeg
def test_retry_exhausted_produces_degraded_chunk(workdir, tmp_path):
    """Engine always fails → every chunk is degraded, job completes."""
    source = tmp_path / "src.png"
    source.write_bytes(_PNG_1x1)
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 200 + b"WAVE")

    engine = FailingEngine(always_fail=True)
    engine.load()
    cfg = _make_chunk_config(retry_max=2, retry_delay=0.0, chunk_seconds=2.0)
    rv = RenderVideo(engine=engine, chunk_config=cfg)
    identity = _mock_identity()
    request = _make_request(audio_path=audio, chunk_seconds=2.0)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "src.application.render_video._probe_audio_duration",
            lambda _: 2.0,
        )
        result = rv.run(request, identity)

    # Exactly one chunk, degraded.
    assert len(result.chunks) == 1
    chunk = result.chunks[0]
    assert "degraded" in chunk.output_path.name
    assert chunk.gpu_seconds == 0.0  # degraded has zero cost
    # Job still completes (no exception).
    assert result.output_path.is_file()


@requires_ffmpeg
def test_mixed_chunks_one_degraded_others_succeed(workdir, tmp_path):
    """Only chunk index 1 fails permanently; chunk 0 succeeds, chunk 1 degraded."""
    source = tmp_path / "src.png"
    source.write_bytes(_PNG_1x1)
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 200 + b"WAVE")

    engine = SelectiveFailingEngine(failing_index=1)
    engine.load()
    cfg = _make_chunk_config(
        retry_max=3, retry_delay=0.0, chunk_seconds=2.0, overlap_seconds=0.0
    )
    rv = RenderVideo(engine=engine, chunk_config=cfg)
    identity = _mock_identity()
    request = _make_request(audio_path=audio, chunk_seconds=2.0)

    # Produce exactly 2 chunks: 4 seconds of audio, 2-second windows.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "src.application.render_video._probe_audio_duration",
            lambda _: 4.0,
        )
        result = rv.run(request, identity)

    assert len(result.chunks) == 2
    # Chunk 0 was rendered successfully.
    chunk0 = result.chunks[0]
    assert "degraded" not in chunk0.output_path.name
    assert chunk0.gpu_seconds > 0
    # Chunk 1 is degraded after exhausting retries.
    chunk1 = result.chunks[1]
    assert "degraded" in chunk1.output_path.name
    assert chunk1.gpu_seconds == 0.0
    # Job still completes.
    assert result.output_path.is_file()
    # Total GPU seconds only counts successful chunks.
    assert result.gpu_seconds_total == chunk0.gpu_seconds


@requires_ffmpeg
def test_retry_attempts_are_exact(workdir, tmp_path):
    """Engine fails exactly on attempts 1 and 2, succeeds on 3 — verify retries."""
    source = tmp_path / "src.png"
    source.write_bytes(_PNG_1x1)
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 200 + b"WAVE")

    call_counter = {"count": 0}

    class CountingEngine(FailingEngine):
        def render_chunk(self, request, identity):
            call_counter["count"] += 1
            return super().render_chunk(request, identity)

    engine = CountingEngine(fail_count=2)
    engine.load()
    cfg = _make_chunk_config(
        retry_max=3, retry_delay=0.0, chunk_seconds=2.0, overlap_seconds=0.0
    )
    rv = RenderVideo(engine=engine, chunk_config=cfg)
    identity = _mock_identity()
    request = _make_request(audio_path=audio, chunk_seconds=2.0)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "src.application.render_video._probe_audio_duration",
            lambda _: 2.0,
        )
        result = rv.run(request, identity)

    assert len(result.chunks) == 1
    assert result.chunks[0].gpu_seconds > 0
    # Exactly 3 calls: 2 failures + 1 success.
    assert call_counter["count"] == 3, (
        f"Expected 3 render_chunk calls (2 fail + 1 success), got {call_counter['count']}"
    )


@requires_ffmpeg
def test_degraded_chunk_output_is_valid_mp4(workdir, tmp_path):
    """Degraded fallback writes a valid mp4 file."""
    source = tmp_path / "src.png"
    source.write_bytes(_PNG_1x1)
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 200 + b"WAVE")

    engine = FailingEngine(always_fail=True)
    engine.load()
    cfg = _make_chunk_config(retry_max=1, retry_delay=0.0, chunk_seconds=2.0)
    rv = RenderVideo(engine=engine, chunk_config=cfg)
    identity = _mock_identity()
    request = _make_request(audio_path=audio, chunk_seconds=2.0)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "src.application.render_video._probe_audio_duration",
            lambda _: 2.0,
        )
        result = rv.run(request, identity)

    degraded = result.chunks[0]
    assert degraded.output_path.is_file(), (
        f"Degraded chunk mp4 not found at {degraded.output_path}"
    )
    assert degraded.output_path.stat().st_size > 0, "Degraded mp4 should be non-empty"


@requires_ffmpeg
def test_retry_respects_custom_chunk_retry_max(workdir, tmp_path):
    """When chunk_retry_max=1, there is no retry — immediate degraded fallback."""
    source = tmp_path / "src.png"
    source.write_bytes(_PNG_1x1)
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 200 + b"WAVE")

    call_counter = {"count": 0}

    class CountingEngine(FailingEngine):
        def render_chunk(self, request, identity):
            call_counter["count"] += 1
            raise RuntimeError("Always fails")

    engine = CountingEngine(always_fail=True)
    engine.load()
    cfg = _make_chunk_config(retry_max=1, retry_delay=0.0, chunk_seconds=2.0)
    rv = RenderVideo(engine=engine, chunk_config=cfg)
    identity = _mock_identity()
    request = _make_request(audio_path=audio, chunk_seconds=2.0)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "src.application.render_video._probe_audio_duration",
            lambda _: 2.0,
        )
        result = rv.run(request, identity)

    assert len(result.chunks) == 1
    assert "degraded" in result.chunks[0].output_path.name
    # With retry_max=1, the engine is called exactly once.
    assert call_counter["count"] == 1, (
        f"Expected 1 render_chunk call, got {call_counter['count']}"
    )
