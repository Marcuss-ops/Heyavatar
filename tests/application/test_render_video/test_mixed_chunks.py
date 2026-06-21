"""Scenario: mixed multi-chunk job — one chunk degrades, others succeed.

Validates that the render loop continues across chunks even after a
degradation event, and that GPU-seconds accounting only includes the
successful chunks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.application.render_video.use_case import RenderVideo
from tests._fixtures import PNG_1X1 as _PNG_1x1

from tests.application.test_render_video._helpers import (
    SelectiveFailingEngine,
    _make_chunk_config,
    _make_request,
    _mock_identity,
    requires_ffmpeg,
)


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
            "src.application.render_video.audio_probe._probe_audio_duration",
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
