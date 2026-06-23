"""Motion benchmark helpers shared by runtime and CLI tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.motion.pose_graph import PoseGraphTrack


@dataclass(slots=True)
class MotionBenchmarkResult:
    gesture_variety: float
    presence_score: float
    motion_energy: float
    transition_density: float
    steady_pose_ratio: float


def _motion_energy(track: PoseGraphTrack) -> float:
    if track.frames == 0:
        return 0.0
    left = np.asarray(track.left_wrist_velocity, dtype=np.float32)
    right = np.asarray(track.right_wrist_velocity, dtype=np.float32)
    if left.size == 0 and right.size == 0:
        return 0.0
    left_score = float(np.mean(np.abs(left))) if left.size else 0.0
    right_score = float(np.mean(np.abs(right))) if right.size else 0.0
    return float(max(left_score, right_score))


def benchmark_pose_track(track: PoseGraphTrack) -> MotionBenchmarkResult:
    states = list(track.pose_state)
    if not states:
        return MotionBenchmarkResult(0.0, 0.0, 0.0, 0.0, 1.0)

    unique_states = {
        state
        for state in states
        if state not in {"neutral_desk", "idle_small", ""}
    }
    gesture_variety = min(1.0, len(unique_states) / 6.0)
    transition_density = min(1.0, len(track.transition_frames) / max(1.0, track.frames / 12.0))
    steady_pose_ratio = float(sum(1 for state in states if state == "neutral_desk") / len(states))
    motion_energy = min(1.0, _motion_energy(track) / 1.5)
    presence_score = float(
        0.34 * gesture_variety + 0.38 * transition_density + 0.28 * motion_energy
    )
    return MotionBenchmarkResult(
        gesture_variety=float(gesture_variety),
        presence_score=float(np.clip(presence_score, 0.0, 1.0)),
        motion_energy=float(motion_energy),
        transition_density=float(transition_density),
        steady_pose_ratio=float(steady_pose_ratio),
    )


def benchmark_pose_track_path(path: Path) -> MotionBenchmarkResult:
    return benchmark_pose_track(PoseGraphTrack.from_npz(path))
