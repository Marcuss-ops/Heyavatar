"""Tests for ``src.motion.text_driven_timeline``.

The module is the glue between the previously disconnected
:class:`RuleBasedGesturePlanner` and the canonical Change 4 timeline
shape consumed by the future ``align_timeline`` step. These tests
exercise the algorithm end-to-end against the real planner + the
real ``registry/gestures.yaml`` registry, with a single caller-
supplied ``planner_factory`` override kept for paranoia.
"""

from __future__ import annotations

from typing import List

import pytest

from contracts.gesture_planner import GestureIntent
from src.motion.text_driven_timeline import (
    Timeline,
    TimelineSegment,
    pick_pose_id_for_gesture,
    text_to_timeline,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


class _StubPlanner:
    """Test planner that returns a configurable list of intents.

    Lets the tests cover the distribution / padding / overshoot logic
    without relying on the keyword-trigger behaviour of the production
    planner. (The production planner is still exercised through the
    real-string integration test at the bottom.)
    """

    def __init__(self, intents: List[GestureIntent]) -> None:
        self._intents = intents

    def plan(self, text: str, avatar_id: str, language: str) -> List[GestureIntent]:
        return list(self._intents)


# ─────────────────────────────────────────────────────────────────────────────
# pick_pose_id_for_gesture
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("gesture_id", "expected_pose"),
    [
        ("point_left", "left_hand_up"),
        ("explain_left", "left_hand_up"),
        ("point_right", "right_hand_up"),
        ("explain_right", "right_hand_up"),
        ("explain_both", "both_hands_open"),
        ("open_palms", "both_hands_open"),
        ("comparison", "both_hands_open"),
        ("idle_small", "neutral_desk"),
        ("unknown_gesture", "neutral_desk"),
    ],
)
def test_pick_pose_id_for_gesture_maps_known_ids(gesture_id: str, expected_pose: str) -> None:
    assert pick_pose_id_for_gesture(gesture_id) == expected_pose


# ─────────────────────────────────────────────────────────────────────────────
# text_to_timeline — degenerate / edge inputs
# ─────────────────────────────────────────────────────────────────────────────


def test_text_to_timeline_with_zero_duration_returns_single_idle() -> None:
    timeline = text_to_timeline(
        text="qualunque cosa",
        audio_duration=0.0,
        planner_factory=lambda: _StubPlanner(
            [
                GestureIntent(
                    text_span="qualunque",
                    gesture_id="count_three",
                    anchor_word="tre",
                    intensity=0.9,
                )
            ]
        ),
    )
    assert timeline.duration == 0.0
    assert len(timeline.segments) == 1
    assert timeline.segments[0].kind == "idle"


def test_text_to_timeline_with_no_intents_returns_single_idle() -> None:
    timeline = text_to_timeline(
        text="nessun gesto qui",
        audio_duration=5.0,
        planner_factory=lambda: _StubPlanner([]),
    )
    assert timeline.duration == 5.0
    assert len(timeline.segments) == 1
    assert timeline.segments[0].kind == "idle"
    assert timeline.segments[0].start == 0.0
    assert timeline.segments[0].end == 5.0


# ─────────────────────────────────────────────────────────────────────────────
# text_to_timeline — duration distribution
# ─────────────────────────────────────────────────────────────────────────────


def test_text_to_timeline_distributes_intents_with_lead_in() -> None:
    text = (
        "Oggi parliamo di tre differenze molto importanti e "
        "alla fine vi spiego come cambiare le impostazioni"
    )
    planner = RuleBasedPlannerSpy()
    timeline = text_to_timeline(
        text=text,
        audio_duration=10.0,
        planner_factory=lambda: planner,
    )

    # The real rule-based planner should emit multiple intents for
    # this text: at least a count, an emphasis, and a conclusion.
    assert planner.call_count == 1
    assert len(timeline.segments) >= 4  # at least 1 idle lead-in + intents + idle tail
    # The very first segment is an idle lead-in (or the first gesture
    # if the anchor is at t=0); verify it is contiguous with t=0.
    assert timeline.segments[0].start == pytest.approx(0.0, abs=1e-3)
    # And every segment is contiguous (no gaps): end[i] == start[i+1]
    for prev, nxt in zip(timeline.segments, timeline.segments[1:]):
        assert nxt.start == pytest.approx(prev.end, abs=1e-3)
    # And the timeline ends exactly at the audio_duration.
    assert timeline.segments[-1].end == pytest.approx(timeline.duration, abs=1e-3)
    # And every gesture segment has an intensity > 0 since the
    # planner only schedules intents with intensities.
    for seg in timeline.segments:
        if seg.kind == "gesture":
            assert 0.0 < seg.intensity <= 1.0
            assert seg.pose_id in {
                "neutral_desk",
                "left_hand_up",
                "right_hand_up",
                "both_hands_open",
            }


def test_text_to_timeline_clips_overshoot_at_audio_boundary() -> None:
    """A single intent with audio_duration <= its duration must be
    clamped so end == audio_duration, not pushed past it."""
    planner = _StubPlanner(
        [
            GestureIntent(
                text_span="count",
                gesture_id="count_three",
                anchor_word="tre",
                intensity=0.9,
            )
        ]
    )
    timeline = text_to_timeline(
        text="tre",
        audio_duration=1.0,
        planner_factory=lambda: planner,
    )
    assert timeline.duration == 1.0
    assert timeline.segments[-1].end <= 1.0


def test_text_to_timeline_drops_gestures_past_audio_end_when_too_short() -> None:
    """Two intents totalling > audio_duration: at least one must be
    clipped/merged. Verify the timeline never exceeds audio_duration
    even when the planner emits overlapping gestures."""
    planner = _StubPlanner(
        [
            GestureIntent(
                text_span="count one",
                gesture_id="count_one",
                anchor_word="uno",
                intensity=0.9,
            ),
            GestureIntent(
                text_span="count three",
                gesture_id="count_three",
                anchor_word="tre",
                intensity=0.9,
            ),
        ]
    )
    timeline = text_to_timeline(
        text="uno tre",
        audio_duration=2.0,  # tight: 2 gestures totalling >6s nominal duration
        planner_factory=lambda: planner,
    )
    assert timeline.duration == 2.0
    for seg in timeline.segments:
        assert seg.start <= timeline.duration
        assert seg.end <= timeline.duration


# ─────────────────────────────────────────────────────────────────────────────
# text_to_timeline — JSON round-trip
# ─────────────────────────────────────────────────────────────────────────────


def test_timeline_round_trip_via_dict() -> None:
    planner = _StubPlanner(
        [
            GestureIntent(
                text_span="anche confronto",
                gesture_id="comparison",
                anchor_word="contro",
                intensity=0.8,
            )
        ]
    )
    timeline = text_to_timeline(
        text="confronto",
        audio_duration=6.0,
        planner_factory=lambda: planner,
    )
    payload = timeline.to_dict()
    assert "duration" in payload
    assert "fps" in payload
    assert "segments" in payload
    assert payload["duration"] == pytest.approx(6.0)

    restored = Timeline.from_dict(payload)
    assert restored.duration == pytest.approx(timeline.duration)
    assert restored.fps == timeline.fps
    assert len(restored.segments) == len(timeline.segments)
    for src, dst in zip(timeline.segments, restored.segments):
        assert dst.kind == src.kind
        assert dst.gesture_id == src.gesture_id
        assert dst.pose_id == src.pose_id
        assert dst.start == pytest.approx(src.start, abs=1e-3)
        assert dst.end == pytest.approx(src.end, abs=1e-3)
        assert dst.intensity == pytest.approx(src.intensity, abs=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# text_to_timeline — real keyword planner integration
# ─────────────────────────────────────────────────────────────────────────────


class RuleBasedPlannerSpy:
    """A spy that forwards to the real RuleBasedGesturePlanner so we
    can introspect call count without mocking the whole class."""

    def __init__(self) -> None:
        from providers.motion_extraction.mediapipe.gesture_planner import (
            RuleBasedGesturePlanner,
        )

        self._inner = RuleBasedGesturePlanner()
        self.call_count = 0

    def plan(self, text: str, avatar_id: str, language: str) -> List[GestureIntent]:
        self.call_count += 1
        return self._inner.plan(text, avatar_id, language)


def test_real_rule_based_planner_produces_known_intents() -> None:
    """End-to-end sanity: the Italian + English keyword tables in the
    real planner still trigger the gesture mapping for count +
    emphasis + comparison + open_palms."""
    timeline = text_to_timeline(
        text="Ciao a tutti, oggi vediamo tre differenze molto importanti tra le opzioni.",
        audio_duration=6.5,
    )
    gestures = [s for s in timeline.segments if s.kind == "gesture"]
    gesture_ids = {s.gesture_id for s in gestures}
    # The real planner picks the first matching keyword in each
    # family; "tre", "molto", "ciao" → count_three, emphasis_small,
    # open_palms → we get at least one of: count_three, emphasis_small,
    # open_palms.
    assert gesture_ids & {"count_three", "emphasis_small", "open_palms"}, (
        f"expected at least one of count_three/emphasis_small/open_palms, got {gesture_ids}"
    )


def test_real_rule_based_planner_invariants_on_long_text() -> None:
    long_text = (
        "Ciao. " * 100
        + " tre differenze molto importanti "
        + "come funziona questa cosa veramente?"
    )
    timeline = text_to_timeline(text=long_text, audio_duration=20.0)
    assert timeline.duration == pytest.approx(20.0)
    # Every consecutive pair must be contiguous.
    for prev, nxt in zip(timeline.segments, timeline.segments[1:]):
        assert nxt.start == pytest.approx(prev.end, abs=1e-3)
    # And every id segment uses the canonical rest pose.
    for seg in timeline.segments:
        if seg.kind == "idle":
            assert seg.pose_id == "neutral_desk"
            assert seg.gesture_id == "idle_small"
