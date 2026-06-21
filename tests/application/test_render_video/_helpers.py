"""Shared helpers and test doubles for RenderVideo retry tests.

* :data:`requires_ffmpeg` — pytest marker that skips when ffmpeg is missing.
* Test-engine factories (:class:`FailingEngine`, :class:`SelectiveFailingEngine`)
  used to inject deterministic failure behaviours.
* Request helpers (mock identity, request object, chunk config, successful chunk).
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from contracts.avatar_engine import AvatarEngine, EngineState
from src.application.render_video.config import ChunkConfig
from src.core.config import get_settings
from src.domain.enums import EngineId
from src.domain.types import (
    AvatarIdentityHandle,
    IdentityId,
    IdentitySpec,
    RenderChunkRequest,
    RenderChunkResult,
    RenderJobId,
    RenderRequest,
    RenderSpec,
    Tier,
)


requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="test shells out to ffmpeg for degraded mp4 stub",
)


# ── request / result factories ──────────────────────────────────────


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


# ── test engines ────────────────────────────────────────────────────


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
