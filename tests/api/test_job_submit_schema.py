from __future__ import annotations

from api.schemas.jobs import JobSubmitRequest


def test_job_submit_request_includes_motion_style_in_payload() -> None:
    payload = JobSubmitRequest(
        identity_id="id-alice",
        source_image="/tmp/source.png",
        audio_path="/tmp/audio.wav",
        motion_style="expressive",
    )

    queue_payload = payload.to_queue_payload()
    assert queue_payload["motion_style"] == "expressive"
