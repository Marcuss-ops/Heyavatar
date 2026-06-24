from __future__ import annotations

from src.motion.face_motion_timeline import text_to_face_motion_timeline


def test_face_motion_timeline_is_contiguous_and_hand_free() -> None:
    timeline = text_to_face_motion_timeline(
        "Abbiamo una conclusione molto importante, vero?",
        audio_duration=6.0,
    )

    assert timeline.duration == 6.0
    assert timeline.segments
    assert timeline.segments[0].start == 0.0
    for prev, nxt in zip(timeline.segments, timeline.segments[1:]):
        assert nxt.start >= prev.end
    assert timeline.segments[-1].end == 6.0
    assert all("hand" not in seg.motion_id for seg in timeline.segments)

