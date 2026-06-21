"""Scenario: ``chunk_retry_max=1`` means no retry — immediate degraded fallback."""

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
            "src.application.render_video.audio_probe._probe_audio_duration",
            lambda _: 2.0,
        )
        result = rv.run(request, identity)

    assert len(result.chunks) == 1
    assert "degraded" in result.chunks[0].output_path.name
    # With retry_max=1, the engine is called exactly once.
    assert call_counter["count"] == 1, (
        f"Expected 1 render_chunk call, got {call_counter['count']}"
    )
