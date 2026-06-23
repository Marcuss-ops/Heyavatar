from __future__ import annotations

import numpy as np
from pathlib import Path

from tools.avatar_assets.benchmark_motion_quality import benchmark_motion_quality, benchmark_pose_track
from src.motion.pose_graph import PoseGraphTrack


def test_benchmark_pose_track_scores_presence_from_motion() -> None:
    track = PoseGraphTrack(
        fps=25.0,
        pose_state=("neutral_desk", "right_hand_rising", "right_hand_up", "right_hand_lowering"),
        transition_frames=(1, 2, 3),
        hold_frames=(2,),
        left_hand_landmarks=np.zeros((4, 21, 3), dtype=np.float32),
        right_hand_landmarks=np.zeros((4, 21, 3), dtype=np.float32),
        left_wrist_velocity=np.zeros(4, dtype=np.float32),
        right_wrist_velocity=np.array([0.0, 0.4, 0.2, 0.5], dtype=np.float32),
    )

    result = benchmark_pose_track(track)

    assert result.presence_score > 0.0
    assert result.gesture_variety > 0.0
    assert result.transition_density > 0.0


def test_benchmark_motion_quality_reads_npz(tmp_path: Path) -> None:
    npz_path = tmp_path / "hand_motion.npz"
    np.savez(
        npz_path,
        fps=np.asarray([25.0], dtype=np.float32),
        pose_state=np.asarray(["neutral_desk", "right_hand_up"], dtype="U32"),
        transition_frames=np.asarray([1], dtype=np.int32),
        hold_frames=np.asarray([1], dtype=np.int32),
        left_hand_landmarks=np.zeros((2, 21, 3), dtype=np.float32),
        right_hand_landmarks=np.zeros((2, 21, 3), dtype=np.float32),
        left_wrist_velocity=np.zeros(2, dtype=np.float32),
        right_wrist_velocity=np.array([0.0, 0.2], dtype=np.float32),
    )

    result = benchmark_motion_quality(motion_path=npz_path)

    assert result.presence_score >= 0.0
    assert result.lip_sync_status == "SKIPPED"
