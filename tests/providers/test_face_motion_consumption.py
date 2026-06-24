from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from providers.liveportrait.adapter._mock import _mock_render_chunk
from src.domain.types import RenderChunkRequest
from src.motion.face_motion_timeline import text_to_face_motion_timeline


requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="mock-mode provider tests shell out to ffmpeg",
)


@requires_ffmpeg
def test_mock_render_chunk_consumes_face_motion_timeline(tmp_path: Path) -> None:
    timeline = text_to_face_motion_timeline(
        "Abbiamo una conclusione molto importante, vero?",
        audio_duration=6.0,
    )
    face_timeline_path = tmp_path / "face_timeline.json"
    timeline.write_json(face_timeline_path)

    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"RIFF0000WAVE")

    request = RenderChunkRequest(
        job_id="job-1",
        audio_window=(0.0, 1.0),
        audio_path=audio,
        fps=25,
        face_motion_timeline_path=face_timeline_path,
    )
    result = _mock_render_chunk(request, 1.0, capture_dir=tmp_path)

    sidecar = result.output_path.with_suffix(".face_motion.json")
    assert sidecar.is_file()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["face_motion"]["segment_count"] == len(timeline.segments)
    assert payload["face_motion"]["motion_ids"]
    assert payload["colour"] == "0x223344"
