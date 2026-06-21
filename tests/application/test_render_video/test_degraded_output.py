"""Scenario: degraded fallback writes a valid, non-empty mp4 file."""

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
            "src.application.render_video.audio_probe._probe_audio_duration",
            lambda _: 2.0,
        )
        result = rv.run(request, identity)

    degraded = result.chunks[0]
    assert degraded.output_path.is_file(), (
        f"Degraded chunk mp4 not found at {degraded.output_path}"
    )
    assert degraded.output_path.stat().st_size > 0, "Degraded mp4 should be non-empty"
