from __future__ import annotations

import numpy as np

from tools.avatar_assets.extract_hand_motion import (
    POSE_BOTH_OPEN,
    POSE_NEUTRAL,
    POSE_RIGHT_LOWERING,
    POSE_RIGHT_RISING,
    POSE_RIGHT_UP,
    build_hand_motion_payload,
)


def _make_hand_series(frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left = np.full((frames, 21, 3), np.nan, dtype=np.float32)
    right = np.full((frames, 21, 3), np.nan, dtype=np.float32)
    left_vis = np.zeros(frames, dtype=np.float32)
    right_vis = np.ones(frames, dtype=np.float32)

    for i in range(frames):
        # Right hand rises, holds, then lowers.
        if i < 5:
            y = 0.82 - i * 0.04
        elif i < 11:
            y = 0.62
        else:
            y = 0.62 + (i - 10) * 0.05
        for j in range(21):
            right[i, j, 0] = 0.65 + 0.01 * j
            right[i, j, 1] = y + 0.005 * j
            right[i, j, 2] = 0.0

        # Left hand stays present but mostly neutral.
        for j in range(21):
            left[i, j, 0] = 0.30 + 0.004 * j
            left[i, j, 1] = 0.85 + 0.002 * j
            left[i, j, 2] = 0.0
        left_vis[i] = 1.0

    return left, right, left_vis, right_vis


def test_build_hand_motion_payload_emits_npz_ready_arrays() -> None:
    left, right, left_vis, right_vis = _make_hand_series(frames=14)
    payload, result = build_hand_motion_payload(
        left_image_landmarks=left,
        right_image_landmarks=right,
        left_visibility=left_vis,
        right_visibility=right_vis,
        fps=25.0,
    )

    assert payload["left_hand_landmarks"].shape == (14, 21, 3)
    assert payload["right_hand_landmarks"].shape == (14, 21, 3)
    assert payload["pose_state"].shape == (14,)
    assert result.frames == 14
    assert result.right_hand_detected_ratio == 1.0
    assert result.start_pose in {POSE_NEUTRAL, POSE_RIGHT_RISING}


def test_build_hand_motion_payload_classifies_rise_hold_return() -> None:
    left, right, left_vis, right_vis = _make_hand_series(frames=14)
    payload, result = build_hand_motion_payload(
        left_image_landmarks=left,
        right_image_landmarks=right,
        left_visibility=left_vis,
        right_visibility=right_vis,
        fps=25.0,
    )

    states = list(payload["pose_state"].tolist())
    assert POSE_RIGHT_RISING in states
    assert POSE_RIGHT_UP in states
    assert POSE_RIGHT_LOWERING in states
    assert result.transition_frames
    assert result.hold_frames
    assert result.end_pose in {POSE_NEUTRAL, POSE_BOTH_OPEN, POSE_RIGHT_LOWERING}
