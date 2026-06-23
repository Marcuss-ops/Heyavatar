from __future__ import annotations

from providers.liveportrait.adapter._render import _motion_style_profile


def test_motion_style_profiles_progress_from_subtle_to_expressive() -> None:
    subtle = _motion_style_profile("subtle", 1.0)
    balanced = _motion_style_profile("balanced", 1.0)
    expressive = _motion_style_profile("expressive", 1.0)

    assert subtle["head_pitch"] < balanced["head_pitch"] < expressive["head_pitch"]
    assert subtle["blink_rate"] < balanced["blink_rate"] < expressive["blink_rate"]
    assert subtle["speech_nod"] < balanced["speech_nod"] < expressive["speech_nod"]


def test_eye_lock_damps_lateral_drift_but_keeps_mouth_motion() -> None:
    unlocked = _motion_style_profile("expressive", 1.0, eye_lock=False)
    locked = _motion_style_profile("expressive", 1.0, eye_lock=True)

    assert locked["head_yaw"] < unlocked["head_yaw"]
    assert locked["head_roll"] < unlocked["head_roll"]
    assert locked["sway_x"] < unlocked["sway_x"]
    assert locked["mouth_boost"] == unlocked["mouth_boost"]
