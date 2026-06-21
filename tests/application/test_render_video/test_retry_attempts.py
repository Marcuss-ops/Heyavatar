"""Scenario: retry call-count is exact (no extra silent calls)."""

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
            "src.application.render_video.audio_probe._probe_audio_duration",
            lambda _: 2.0,
        )
        result = rv.run(request, identity)

    assert len(result.chunks) == 1
    assert result.chunks[0].gpu_seconds > 0
    # Exactly 3 calls: 2 failures + 1 success.
    assert call_counter["count"] == 3, (
        f"Expected 3 render_chunk calls (2 fail + 1 success), got {call_counter['count']}"
    )
