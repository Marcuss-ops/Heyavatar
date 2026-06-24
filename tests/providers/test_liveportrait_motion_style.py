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


def test_micro_saccades_in_driving_keypoints() -> None:
    import numpy as np
    import torch
    from providers.liveportrait.audio_bridge.types import DrivingSignals
    from providers.liveportrait.adapter._render import _build_driving_keypoints

    frames = 50
    # Create empty driving signals
    driving = DrivingSignals(
        frames=frames,
        exp_d_flat=np.zeros(frames * 21 * 3, dtype=np.float32),
        blink_mask=np.zeros(frames, dtype=np.float32),
        mouth_aperture=np.zeros(frames, dtype=np.float32),
        backend="mock",
    )
    kp_s = torch.zeros((1, 21, 3), dtype=torch.float32)

    # Call _build_driving_keypoints under legacy/mock fallback (wrapper=None)
    kp_d = _build_driving_keypoints(
        driving=driving,
        kp_s=kp_s,
        torch=torch,
        device="cpu",
        wrapper=None,
        static_head=True,
    )

    # kp_d has shape [50, 21, 3]
    assert kp_d.shape == (frames, 21, 3)

    # Eye keypoints [11, 13, 15, 16, 18] should have saccadic variation
    for eye_idx in [11, 13, 15, 16, 18]:
        # Variance of eye coordinate changes should be > 0 due to saccades
        assert np.var(kp_d[:, eye_idx, 0]) > 0.0 or np.var(kp_d[:, eye_idx, 1]) > 0.0

    # Non-eye keypoints (e.g. index 5) should remain perfectly static (0.0) since exp_d_flat is 0
    assert np.allclose(kp_d[:, 5, :], 0.0)

