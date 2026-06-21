"""Scenario: engine fails N times, succeeds on the (N+1)-th try.

Validates the happy retry path: the chunk is rendered with non-degraded
output and the GPU-seconds telemetry is published for the successful
attempt.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.application.render_video.use_case import RenderVideo
from src.application.telemetry import TelemetryRecorder
from tests._fixtures import PNG_1X1 as _PNG_1x1

from tests.application.test_render_video._helpers import (
    FailingEngine,
    _make_chunk_config,
    _make_request,
    _mock_identity,
    requires_ffmpeg,
)


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
            "src.application.render_video.audio_probe._probe_audio_duration",
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
