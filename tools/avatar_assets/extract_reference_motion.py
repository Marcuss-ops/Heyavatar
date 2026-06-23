"""One-shot CLI that extracts both hand and body motion from a reference video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.avatar_assets.extract_body_motion import extract_body_motion, _write_registry as _write_body_registry
from tools.avatar_assets.extract_hand_motion import extract_hand_motion, _write_registry as _write_hand_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract hand_motion.npz and body_and_hands_motion.npz together.")
    parser.add_argument("--input", required=True, type=Path, help="Reference video path")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for extracted motion assets")
    return parser


def extract_reference_motion(input_path: Path, output_dir: Path) -> dict[str, dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    hand_npz = output_dir / "hand_motion.npz"
    body_npz = output_dir / "body_and_hands_motion.npz"

    hand_summary = extract_hand_motion(input_path, hand_npz)
    body_summary = extract_body_motion(input_path, body_npz)

    _write_hand_registry(output_dir / "hand_poses.yaml")
    _write_body_registry(output_dir / "body_pose_graph.yaml")
    return {
        "hand_motion": {
            "path": str(hand_npz),
            "summary": {
                "fps": hand_summary.fps,
                "frames": hand_summary.frames,
                "left_hand_detected_ratio": hand_summary.left_hand_detected_ratio,
                "right_hand_detected_ratio": hand_summary.right_hand_detected_ratio,
                "start_pose": hand_summary.start_pose,
                "end_pose": hand_summary.end_pose,
            },
        },
        "body_motion": {
            "path": str(body_npz),
            "summary": {
                "fps": body_summary.fps,
                "frames": body_summary.frames,
                "pose_detected_ratio": body_summary.pose_detected_ratio,
                "left_hand_detected_ratio": body_summary.left_hand_detected_ratio,
                "right_hand_detected_ratio": body_summary.right_hand_detected_ratio,
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = extract_reference_motion(args.input, args.output_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
