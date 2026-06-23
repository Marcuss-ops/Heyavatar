"""Extract normalized hand motion from a reference video.

Pipeline:
    video.mp4 -> MediaPipe Hands -> 21 landmarks per hand per frame
    -> temporal smoothing -> normalization -> pose segmentation
    -> hand_motion.npz + hand_motion.json

The tool is intentionally asset-oriented: it captures motion trajectory
and timing, not pixels. Those trajectories can then be re-applied to
new avatar clips with a body/pose graph.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import yaml


HAND_LANDMARK_COUNT = 21
LEFT_HAND = "left"
RIGHT_HAND = "right"
POSE_NEUTRAL = "neutral_desk"
POSE_RIGHT_RISING = "right_hand_rising"
POSE_RIGHT_UP = "right_hand_up"
POSE_RIGHT_LOWERING = "right_hand_lowering"
POSE_LEFT_RISING = "left_hand_rising"
POSE_LEFT_UP = "left_hand_up"
POSE_LEFT_LOWERING = "left_hand_lowering"
POSE_BOTH_OPEN = "both_hands_open"

_HAND_LABELS = (
    LEFT_HAND,
    RIGHT_HAND,
)


@dataclass(slots=True)
class HandMotionResult:
    fps: float
    frames: int
    left_hand_detected_ratio: float
    right_hand_detected_ratio: float
    start_pose: str
    end_pose: str
    peak_right_wrist_frame: int
    transition_frames: list[int]
    hold_frames: list[int]


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
    if fps <= 0.0:
        fps = 25.0
    return frames, fps


def _try_make_mp_hands():
    try:
        import mediapipe as mp  # type: ignore
    except Exception as exc:  # pragma: no cover - import gated in tests
        raise RuntimeError(
            "MediaPipe is required for hand extraction. Install `mediapipe` "
            "or run the extractor on a workstation that has it available."
        ) from exc

    if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "hands"):
        raise RuntimeError("MediaPipe Hands solution API is unavailable in this environment.")

    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _extract_raw_landmarks(
    frames: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return raw hand landmarks and confidence arrays.

    Output shapes:
        left/right image landmarks: (frames, 21, 3)
        left/right visibility: (frames,)
    """
    detector = _try_make_mp_hands()
    left = np.full((len(frames), HAND_LANDMARK_COUNT, 3), np.nan, dtype=np.float32)
    right = np.full_like(left, np.nan)
    left_visibility = np.zeros(len(frames), dtype=np.float32)
    right_visibility = np.zeros(len(frames), dtype=np.float32)

    try:
        for idx, frame_bgr in enumerate(frames):
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)
            if not results.multi_hand_landmarks:
                continue

            hand_landmarks = list(results.multi_hand_landmarks)
            handedness = list(getattr(results, "multi_handedness", []) or [])
            for hand_idx, hand in enumerate(hand_landmarks):
                label = None
                if hand_idx < len(handedness) and handedness[hand_idx].classification:
                    label = handedness[hand_idx].classification[0].label.strip().lower()
                    if label.startswith("left"):
                        label = LEFT_HAND
                    elif label.startswith("right"):
                        label = RIGHT_HAND

                if label not in _HAND_LABELS:
                    xs = [lm.x for lm in hand.landmark]
                    label = LEFT_HAND if float(np.mean(xs)) < 0.5 else RIGHT_HAND

                target = left if label == LEFT_HAND else right
                visibility = left_visibility if label == LEFT_HAND else right_visibility
                score = 1.0
                if hand_idx < len(handedness) and handedness[hand_idx].classification:
                    score = float(handedness[hand_idx].classification[0].score or 1.0)
                for lm_idx, lm in enumerate(hand.landmark[:HAND_LANDMARK_COUNT]):
                    target[idx, lm_idx, 0] = float(lm.x)
                    target[idx, lm_idx, 1] = float(lm.y)
                    target[idx, lm_idx, 2] = float(getattr(lm, "z", 0.0))
                visibility[idx] = score
    finally:
        detector.close()

    return left, right, left_visibility, right_visibility


def _fill_nan_forward_backward(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    for coord in range(out.shape[1]):
        for dim in range(out.shape[2]):
            series = out[:, coord, dim]
            finite = np.isfinite(series)
            if not finite.any():
                continue
            idx = np.flatnonzero(finite)
            series[: idx[0]] = series[idx[0]]
            series[idx[-1] + 1 :] = series[idx[-1]]
            missing = ~finite
            if missing.any():
                series[missing] = np.interp(np.flatnonzero(missing), idx, series[idx])
            out[:, coord, dim] = series
    return out


def _smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=np.float32) / float(window)
    padded = np.pad(values, ((window // 2, window - 1 - window // 2), (0, 0), (0, 0)), mode="edge")
    smoothed = np.empty_like(values)
    for coord in range(values.shape[1]):
        for dim in range(values.shape[2]):
            smoothed[:, coord, dim] = np.convolve(padded[:, coord, dim], kernel, mode="valid")
    return smoothed


def _normalize_hand_landmarks(image_landmarks: np.ndarray) -> np.ndarray:
    """Return wrist-centered, scale-normalized hand landmarks."""
    normalized = np.full_like(image_landmarks, np.nan)
    for idx, frame in enumerate(image_landmarks):
        if not np.isfinite(frame[0, 0]):
            continue
        wrist_xy = frame[0, :2]
        scale_points = []
        for mcp_idx in (5, 9, 17):
            if np.isfinite(frame[mcp_idx, 0]):
                scale_points.append(float(np.linalg.norm(frame[mcp_idx, :2] - wrist_xy)))
        scale = max(float(np.mean(scale_points)) if scale_points else 0.0, 1e-6)
        normalized[idx, :, :2] = (frame[:, :2] - wrist_xy[None, :]) / scale
        normalized[idx, :, 2] = frame[:, 2] / scale
    return normalized


def _signed_wrist_velocity(image_landmarks: np.ndarray, fps: float) -> np.ndarray:
    wrist_y = image_landmarks[:, 0, 1]
    velocity = np.zeros_like(wrist_y, dtype=np.float32)
    finite = np.isfinite(wrist_y)
    if finite.sum() < 2:
        return velocity
    filled = wrist_y.copy()
    finite_idx = np.flatnonzero(finite)
    filled[: finite_idx[0]] = filled[finite_idx[0]]
    filled[finite_idx[-1] + 1 :] = filled[finite_idx[-1]]
    if (~finite).any():
        filled[~finite] = np.interp(np.flatnonzero(~finite), finite_idx, filled[finite_idx])
    velocity[1:] = np.diff(filled) * float(fps)
    return velocity


def _frame_state(
    left_visible: bool,
    right_visible: bool,
    left_y: float,
    right_y: float,
    left_v: float,
    right_v: float,
) -> str:
    threshold = 0.04
    up_y = 0.68
    neutral_y = 0.82

    if left_visible and right_visible and left_y < up_y and right_y < up_y and abs(left_v) <= threshold and abs(right_v) <= threshold:
        return POSE_BOTH_OPEN
    if right_visible:
        if right_v < -threshold and right_y < neutral_y:
            return POSE_RIGHT_RISING
        if right_y < up_y and abs(right_v) <= threshold:
            return POSE_RIGHT_UP
        if right_v > threshold and right_y < neutral_y:
            return POSE_RIGHT_LOWERING
    if left_visible:
        if left_v < -threshold and left_y < neutral_y:
            return POSE_LEFT_RISING
        if left_y < up_y and abs(left_v) <= threshold:
            return POSE_LEFT_UP
        if left_v > threshold and left_y < neutral_y:
            return POSE_LEFT_LOWERING
    if left_visible or right_visible:
        return POSE_NEUTRAL
    return POSE_NEUTRAL


def _segment_frames(states: Sequence[str]) -> tuple[list[tuple[str, int, int]], list[int], list[int]]:
    if not states:
        return [], [], []
    segments: list[tuple[str, int, int]] = []
    start = 0
    current = states[0]
    for idx, state in enumerate(states[1:], start=1):
        if state != current:
            segments.append((current, start, idx))
            current = state
            start = idx
    segments.append((current, start, len(states)))

    transition_frames = [end for _, _, end in segments[:-1]]
    hold_frames = [start for state, start, end in segments if state.endswith("_up") or state == POSE_BOTH_OPEN]
    return segments, transition_frames, hold_frames


def build_hand_motion_payload(
    *,
    left_image_landmarks: np.ndarray,
    right_image_landmarks: np.ndarray,
    left_visibility: np.ndarray,
    right_visibility: np.ndarray,
    fps: float,
) -> tuple[dict[str, np.ndarray], HandMotionResult]:
    left_image_landmarks = _smooth_series(_fill_nan_forward_backward(left_image_landmarks), window=5)
    right_image_landmarks = _smooth_series(_fill_nan_forward_backward(right_image_landmarks), window=5)

    left_normalized = _normalize_hand_landmarks(left_image_landmarks)
    right_normalized = _normalize_hand_landmarks(right_image_landmarks)

    left_wrist_v = _signed_wrist_velocity(left_image_landmarks, fps)
    right_wrist_v = _signed_wrist_velocity(right_image_landmarks, fps)

    states: list[str] = []
    for idx in range(left_image_landmarks.shape[0]):
        left_visible = bool(np.isfinite(left_image_landmarks[idx, 0, 0]) and left_visibility[idx] > 0.0)
        right_visible = bool(np.isfinite(right_image_landmarks[idx, 0, 0]) and right_visibility[idx] > 0.0)
        left_y = float(left_image_landmarks[idx, 0, 1]) if left_visible else 1.0
        right_y = float(right_image_landmarks[idx, 0, 1]) if right_visible else 1.0
        states.append(
            _frame_state(
                left_visible=left_visible,
                right_visible=right_visible,
                left_y=left_y,
                right_y=right_y,
                left_v=float(left_wrist_v[idx]),
                right_v=float(right_wrist_v[idx]),
            )
        )

    segments, transition_frames, hold_frames = _segment_frames(states)
    right_valid = np.isfinite(right_image_landmarks[:, 0, 0])
    left_valid = np.isfinite(left_image_landmarks[:, 0, 0])
    peak_right = int(np.nanargmin(right_image_landmarks[:, 0, 1])) if right_valid.any() else 0

    payload = {
        "fps": np.asarray([fps], dtype=np.float32),
        "timestamp_ms": np.round(np.arange(left_image_landmarks.shape[0]) * (1000.0 / fps)).astype(np.int32),
        "left_hand_image_landmarks": left_image_landmarks.astype(np.float32),
        "right_hand_image_landmarks": right_image_landmarks.astype(np.float32),
        "left_hand_landmarks": left_normalized.astype(np.float32),
        "right_hand_landmarks": right_normalized.astype(np.float32),
        "left_visibility": left_visibility.astype(np.float32),
        "right_visibility": right_visibility.astype(np.float32),
        "left_wrist_velocity": left_wrist_v.astype(np.float32),
        "right_wrist_velocity": right_wrist_v.astype(np.float32),
        "pose_state": np.asarray(states, dtype="U32"),
        "pose_segments": np.asarray([f"{state}:{start}:{end}" for state, start, end in segments], dtype="U64"),
    }

    result = HandMotionResult(
        fps=fps,
        frames=left_image_landmarks.shape[0],
        left_hand_detected_ratio=float(left_valid.mean()) if left_valid.size else 0.0,
        right_hand_detected_ratio=float(right_valid.mean()) if right_valid.size else 0.0,
        start_pose=states[0] if states else POSE_NEUTRAL,
        end_pose=states[-1] if states else POSE_NEUTRAL,
        peak_right_wrist_frame=peak_right,
        transition_frames=transition_frames,
        hold_frames=hold_frames,
    )
    return payload, result


def extract_hand_motion(video_path: Path, output_path: Path, *, smoothing_window: int = 5) -> HandMotionResult:
    frames, fps = _read_video_frames(video_path)
    if not frames:
        raise RuntimeError(f"No frames found in {video_path}")
    left_raw, right_raw, left_vis, right_vis = _extract_raw_landmarks(frames)
    payload, result = build_hand_motion_payload(
        left_image_landmarks=left_raw,
        right_image_landmarks=right_raw,
        left_visibility=left_vis,
        right_visibility=right_vis,
        fps=fps,
    )

    # Re-smooth with caller-controlled window when needed.
    if smoothing_window != 5:
        payload["left_hand_image_landmarks"] = _smooth_series(
            _fill_nan_forward_backward(left_raw), window=smoothing_window
        ).astype(np.float32)
        payload["right_hand_image_landmarks"] = _smooth_series(
            _fill_nan_forward_backward(right_raw), window=smoothing_window
        ).astype(np.float32)
        payload["left_hand_landmarks"] = _normalize_hand_landmarks(
            payload["left_hand_image_landmarks"]
        ).astype(np.float32)
        payload["right_hand_landmarks"] = _normalize_hand_landmarks(
            payload["right_hand_image_landmarks"]
        ).astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)

    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "fps": result.fps,
                "frames": result.frames,
                "left_hand_detected_ratio": result.left_hand_detected_ratio,
                "right_hand_detected_ratio": result.right_hand_detected_ratio,
                "start_pose": result.start_pose,
                "end_pose": result.end_pose,
                "peak_right_wrist_frame": result.peak_right_wrist_frame,
                "transition_frames": result.transition_frames,
                "hold_frames": result.hold_frames,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


def _write_registry(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "poses": {
            POSE_NEUTRAL: {
                "both_hands_visible": True,
                "right_wrist_height_range": [0.0, 0.08],
                "left_wrist_height_range": [0.0, 0.08],
            },
            POSE_RIGHT_RISING: {
                "moving_hand": RIGHT_HAND,
                "state": "rise",
                "duration_seconds": [0.45, 0.85],
            },
            POSE_RIGHT_UP: {
                "moving_hand": RIGHT_HAND,
                "state": "hold",
                "duration_seconds": [0.8, 2.0],
            },
            POSE_RIGHT_LOWERING: {
                "moving_hand": RIGHT_HAND,
                "state": "return",
                "duration_seconds": [0.45, 0.85],
            },
            POSE_LEFT_RISING: {
                "moving_hand": LEFT_HAND,
                "state": "rise",
                "duration_seconds": [0.45, 0.85],
            },
            POSE_LEFT_UP: {
                "moving_hand": LEFT_HAND,
                "state": "hold",
                "duration_seconds": [0.8, 2.0],
            },
            POSE_LEFT_LOWERING: {
                "moving_hand": LEFT_HAND,
                "state": "return",
                "duration_seconds": [0.45, 0.85],
            },
            POSE_BOTH_OPEN: {
                "moving_hand": "both",
                "state": "hold",
                "duration_seconds": [0.8, 2.5],
            },
        },
        "transitions": {
            "neutral_to_right_up": {
                "moving_hand": RIGHT_HAND,
                "min_duration_seconds": 0.45,
                "max_duration_seconds": 0.85,
            },
            "right_up_to_neutral": {
                "moving_hand": RIGHT_HAND,
                "min_duration_seconds": 0.45,
                "max_duration_seconds": 0.85,
            },
            "neutral_to_left_up": {
                "moving_hand": LEFT_HAND,
                "min_duration_seconds": 0.45,
                "max_duration_seconds": 0.85,
            },
            "left_up_to_neutral": {
                "moving_hand": LEFT_HAND,
                "min_duration_seconds": 0.45,
                "max_duration_seconds": 0.85,
            },
        },
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract normalized hand motion from a reference video.")
    parser.add_argument("--input", required=True, type=Path, help="Reference video path")
    parser.add_argument("--output", required=True, type=Path, help="Output hand_motion.npz path")
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
        help="Temporal smoothing window in frames (default: 5)",
    )
    parser.add_argument(
        "--write-registry",
        type=Path,
        default=Path("registry/hand_poses.yaml"),
        help="Write a baseline hand pose registry here",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    result = extract_hand_motion(
        args.input,
        args.output,
        smoothing_window=max(1, int(args.smoothing_window)),
    )
    _write_registry(args.write_registry)

    print(json.dumps(
        {
            "fps": result.fps,
            "frames": result.frames,
            "left_hand_detected_ratio": result.left_hand_detected_ratio,
            "right_hand_detected_ratio": result.right_hand_detected_ratio,
            "start_pose": result.start_pose,
            "end_pose": result.end_pose,
            "peak_right_wrist_frame": result.peak_right_wrist_frame,
            "transition_frames": result.transition_frames,
            "hold_frames": result.hold_frames,
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
