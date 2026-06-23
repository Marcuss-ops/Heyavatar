from __future__ import annotations

from providers.motion_extraction.mediapipe.gesture_planner import RuleBasedGesturePlanner


def test_rule_based_gesture_planner_prefers_count_and_emphasis() -> None:
    planner = RuleBasedGesturePlanner()
    intents = planner.plan("Abbiamo tre cose molto importanti da vedere", "avatar-1", "it")

    assert intents[0].gesture_id == "count_three"
    assert any(intent.gesture_id == "emphasis_small" for intent in intents)


def test_rule_based_gesture_planner_emits_open_palms_for_greeting() -> None:
    planner = RuleBasedGesturePlanner()
    intents = planner.plan("Ciao e benvenuti oggi", "avatar-1", "it")

    assert intents[0].gesture_id in {"open_palms", "idle_small"}
