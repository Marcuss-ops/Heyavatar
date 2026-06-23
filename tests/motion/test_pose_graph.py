from __future__ import annotations

import numpy as np
from pathlib import Path

from src.motion.pose_graph import PoseGraphTrack


def test_pose_graph_track_loads_npz(tmp_path: Path) -> None:
    npz_path = tmp_path / "hand_motion.npz"
    np.savez(
        npz_path,
        fps=np.asarray([25.0], dtype=np.float32),
        pose_state=np.asarray(["neutral_desk", "right_hand_rising", "right_hand_up"], dtype="U32"),
        transition_frames=np.asarray([1, 2], dtype=np.int32),
        hold_frames=np.asarray([2], dtype=np.int32),
        left_hand_landmarks=np.zeros((3, 21, 3), dtype=np.float32),
        right_hand_landmarks=np.zeros((3, 21, 3), dtype=np.float32),
        left_wrist_velocity=np.zeros(3, dtype=np.float32),
        right_wrist_velocity=np.zeros(3, dtype=np.float32),
    )

    track = PoseGraphTrack.from_npz(npz_path)

    assert track.frames == 3
    assert track.pose_at(1) == "right_hand_rising"
    assert track.active_segment_bounds() == (1, 2)
    assert len(track.iter_state_runs()) == 3
