"""Extract normalized body + hand motion from a reference video.

This is the companion to ``extract_hand_motion.py``. It stores reusable
trajectory data for shoulders, elbows, wrists, and hands so we can build
gesture timing assets without copying pixels from the source video.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


BODY_LANDMARK_COUNT = 33
HAND_LANDMARK_COUNT = 21


@dataclass(slots=True)
class BodyMotionSummary:
    fps: float
    frames: int
    pose_detected_ratio: float
    left_hand_detected_ratio: float
    right_hand_detected_ratio: float
    shoulder_motion_mean: float
    wrist_motion_mean: float


def _read_video_frames(video_path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps if fps > 0 else 25.0


def _try_make_holistic():
    try:
        import mediapipe as mp  # type: ignore
    except Exception as exc:  # pragma: no cover - import gated in tests
        raise RuntimeError(
            "MediaPipe is required for body extraction. Install `mediapipe` "
            "to run the extractor."
        ) from exc

    if not hasattr(mp, "solutions"):
        raise RuntimeError("MediaPipe solution API unavailable.")
    return mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _mean_speed(series: np.ndarray, fps: float) -> float:
    if len(series) < 2:
        return 0.0
    diff = np.diff(series, axis=0)
    norms = np.linalg.norm(diff, axis=1)
    return float(np.mean(norms) * fps)


def extract_body_motion(video_path: Path, output_path: Path) -> BodyMotionSummary:
    frames, fps = _read_video_frames(video_path)
    if not frames:
        raise RuntimeError(f"No frames found in {video_path}")

    detector = _try_make_holistic()
    pose_landmarks = np.full((len(frames), BODY_LANDMARK_COUNT, 4), np.nan, dtype=np.float32)
    left_hand = np.full((len(frames), HAND_LANDMARK_COUNT, 4), np.nan, dtype=np.float32)
    right_hand = np.full_like(left_hand, np.nan)
    pose_visible = np.zeros(len(frames), dtype=np.float32)
    left_visible = np.zeros(len(frames), dtype=np.float32)
    right_visible = np.zeros(len(frames), dtype=np.float32)

    try:
        for idx, frame_bgr in enumerate(frames):
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)
            if results.pose_landmarks:
                pose_visible[idx] = 1.0
                for lm_idx, lm in enumerate(results.pose_landmarks.landmark[:BODY_LANDMARK_COUNT]):
                    pose_landmarks[idx, lm_idx, 0] = float(lm.x)
                    pose_landmarks[idx, lm_idx, 1] = float(lm.y)
                    pose_landmarks[idx, lm_idx, 2] = float(getattr(lm, "z", 0.0))
                    pose_landmarks[idx, lm_idx, 3] = float(getattr(lm, "visibility", 0.0))
            if results.left_hand_landmarks:
                left_visible[idx] = 1.0
                for lm_idx, lm in enumerate(results.left_hand_landmarks.landmark[:HAND_LANDMARK_COUNT]):
                    left_hand[idx, lm_idx, 0] = float(lm.x)
                    left_hand[idx, lm_idx, 1] = float(lm.y)
                    left_hand[idx, lm_idx, 2] = float(getattr(lm, "z", 0.0))
                    left_hand[idx, lm_idx, 3] = 1.0
            if results.right_hand_landmarks:
                right_visible[idx] = 1.0
                for lm_idx, lm in enumerate(results.right_hand_landmarks.landmark[:HAND_LANDMARK_COUNT]):
                    right_hand[idx, lm_idx, 0] = float(lm.x)
                    right_hand[idx, lm_idx, 1] = float(lm.y)
                    right_hand[idx, lm_idx, 2] = float(getattr(lm, "z", 0.0))
                    right_hand[idx, lm_idx, 3] = 1.0
    finally:
        detector.close()

    shoulder_idx = [11, 12]
    wrist_idx = [15, 16]
    shoulder_motion = _mean_speed(pose_landmarks[:, shoulder_idx, :2].mean(axis=1), fps)
    wrist_motion = _mean_speed(pose_landmarks[:, wrist_idx, :2].mean(axis=1), fps)

    payload = {
        "fps": np.asarray([fps], dtype=np.float32),
        "timestamp_ms": np.round(np.arange(len(frames)) * (1000.0 / fps)).astype(np.int32),
        "pose_landmarks": pose_landmarks,
        "left_hand_landmarks": left_hand,
        "right_hand_landmarks": right_hand,
        "pose_visible": pose_visible,
        "left_hand_visible": left_visible,
        "right_hand_visible": right_visible,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)

    summary = BodyMotionSummary(
        fps=fps,
        frames=len(frames),
        pose_detected_ratio=float(pose_visible.mean()),
        left_hand_detected_ratio=float(left_visible.mean()),
        right_hand_detected_ratio=float(right_visible.mean()),
        shoulder_motion_mean=shoulder_motion,
        wrist_motion_mean=wrist_motion,
    )

    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "fps": summary.fps,
                "frames": summary.frames,
                "pose_detected_ratio": summary.pose_detected_ratio,
                "left_hand_detected_ratio": summary.left_hand_detected_ratio,
                "right_hand_detected_ratio": summary.right_hand_detected_ratio,
                "shoulder_motion_mean": summary.shoulder_motion_mean,
                "wrist_motion_mean": summary.wrist_motion_mean,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def _write_registry(path: Path) -> None:
    data = {
        "body_states": {
            "neutral_desk": {
                "shoulders_level": True,
                "wrist_height_range": [0.0, 0.08],
            },
            "speech_presenter": {
                "shoulders_level": True,
                "wrist_height_range": [0.08, 0.24],
            },
        },
        "gesture_states": {
            "right_hand_rising": {"moving_hand": "right", "phase": "rise"},
            "right_hand_up": {"moving_hand": "right", "phase": "hold"},
            "right_hand_lowering": {"moving_hand": "right", "phase": "return"},
            "left_hand_rising": {"moving_hand": "left", "phase": "rise"},
            "left_hand_up": {"moving_hand": "left", "phase": "hold"},
            "left_hand_lowering": {"moving_hand": "left", "phase": "return"},
            "both_hands_open": {"moving_hand": "both", "phase": "hold"},
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract reusable body and hand motion from a reference video.")
    parser.add_argument("--input", required=True, type=Path, help="Reference video path")
    parser.add_argument("--output", required=True, type=Path, help="Output body_and_hands_motion.npz path")
    parser.add_argument(
        "--write-registry",
        type=Path,
        default=Path("registry/body_pose_graph.yaml"),
        help="Write the baseline body pose graph registry here",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = extract_body_motion(args.input, args.output)
    _write_registry(args.write_registry)
    print(
        json.dumps(
            {
                "fps": summary.fps,
                "frames": summary.frames,
                "pose_detected_ratio": summary.pose_detected_ratio,
                "left_hand_detected_ratio": summary.left_hand_detected_ratio,
                "right_hand_detected_ratio": summary.right_hand_detected_ratio,
                "shoulder_motion_mean": summary.shoulder_motion_mean,
                "wrist_motion_mean": summary.wrist_motion_mean,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
