from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.avatar_assets import extract_reference_motion as erm


def test_extract_reference_motion_combines_outputs(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "reference.mp4"
    input_path.write_bytes(b"fake")

    hand_npz = tmp_path / "out" / "hand_motion.npz"
    body_npz = tmp_path / "out" / "body_and_hands_motion.npz"

    class _HandSummary:
        fps = 25.0
        frames = 10
        left_hand_detected_ratio = 1.0
        right_hand_detected_ratio = 1.0
        start_pose = "neutral_desk"
        end_pose = "right_hand_up"

    class _BodySummary:
        fps = 25.0
        frames = 10
        pose_detected_ratio = 1.0
        left_hand_detected_ratio = 1.0
        right_hand_detected_ratio = 1.0

    monkeypatch.setattr(erm, "extract_hand_motion", lambda *_a, **_k: _HandSummary())
    monkeypatch.setattr(erm, "extract_body_motion", lambda *_a, **_k: _BodySummary())

    result = erm.extract_reference_motion(input_path, tmp_path / "out")

    assert Path(result["hand_motion"]["path"]) == hand_npz
    assert Path(result["body_motion"]["path"]) == body_npz
    assert (tmp_path / "out" / "hand_poses.yaml").is_file()
    assert (tmp_path / "out" / "body_pose_graph.yaml").is_file()
