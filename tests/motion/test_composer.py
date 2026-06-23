from __future__ import annotations

from contracts.gesture_planner import GestureIntent
from src.motion.composer import MotionTimelineComposer


def test_motion_timeline_composer_adds_idle_and_gesture_segments() -> None:
    composer = MotionTimelineComposer()
    intents = [
        GestureIntent(
            text_span="tre cose",
            gesture_id="count_three",
            anchor_word="tre",
            intensity=0.9,
        )
    ]
    timeline = composer.compose(
        intents,
        [
            {"word": "oggi", "start": 0.0, "end": 0.25},
            {"word": "tre", "start": 1.2, "end": 1.4},
        ],
    )

    assert timeline.segments[0].kind == "idle"
    assert timeline.segments[1].kind == "gesture"
    assert timeline.segments[1].pose_id == "neutral_desk"
    assert timeline.segments[1].stroke_time == 1.2
    assert timeline.duration > 0.0
