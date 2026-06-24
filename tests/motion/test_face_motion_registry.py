from __future__ import annotations

from src.motion.face_planner import RuleBasedFaceMotionPlanner
from src.motion.face_registry import FaceMotionRegistry


def test_face_motion_registry_loads_hand_free_set() -> None:
    registry = FaceMotionRegistry()

    assert "face_idle_soft" in registry.motions
    assert "question_face" in registry.motions
    assert registry.get_motion("smile_small").requires_hands is False
    assert len(registry.motions) == 6
    assert all(not motion.requires_hands for motion in registry.list_hand_free())


def test_face_motion_planner_prefers_face_only_cues() -> None:
    planner = RuleBasedFaceMotionPlanner()

    intents = planner.plan(
        "Abbiamo una conclusione molto importante, vero?",
        "avatar-1",
        "it",
    )

    motion_ids = [intent.motion_id for intent in intents]
    assert "question_face" in motion_ids
    assert "brow_raise_small" in motion_ids
    assert "nod_small" in motion_ids
    assert all("hand" not in motion_id for motion_id in motion_ids)
