"""Scenario: retry budget exhausted -> degraded chunk fallback.

Verifies that when the engine fails every attempt, the chunk is
replaced by a degraded mp4 (zero GPU-seconds, red colour stub) so the
job can still complete without crashing the worker.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.application.render_video.use_case import RenderVideo
from tests._fixtures import PNG_1X1 as _PNG_1x1

from tests.application.test_render_video._helpers import (
    FailingEngine,
    _make_chunk_config,
    _make_request,
    _mock_identity,
    requires_ffmpeg,
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
            "src.application.render_video.audio_probe._probe_audio_duration",
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
