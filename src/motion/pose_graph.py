"""Runtime helpers for reusable pose and hand-motion graphs.

The extractor tools write normalized motion assets to ``*.npz`` files.
This module provides a tiny runtime wrapper that can load those assets
and answer questions like "what pose is active at frame 37?" or
"where are the gesture transitions?" without caring about pixel space.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(slots=True, frozen=True)
class PoseGraphTrack:
    fps: float
    pose_state: tuple[str, ...]
    transition_frames: tuple[int, ...]
    hold_frames: tuple[int, ...]
    left_hand_landmarks: np.ndarray
    right_hand_landmarks: np.ndarray
    left_wrist_velocity: np.ndarray
    right_wrist_velocity: np.ndarray

    @classmethod
    def from_npz(cls, path: Path) -> "PoseGraphTrack":
        data = np.load(path, allow_pickle=True)
        pose_state = tuple(str(v) for v in data.get("pose_state", []))
        transition_frames = tuple(int(v) for v in data.get("transition_frames", []))
        hold_frames = tuple(int(v) for v in data.get("hold_frames", []))
        fps = float(np.asarray(data.get("fps", np.asarray([25.0]))).reshape(-1)[0])
        return cls(
            fps=fps,
            pose_state=pose_state,
            transition_frames=transition_frames,
            hold_frames=hold_frames,
            left_hand_landmarks=np.asarray(
                data.get("left_hand_landmarks", np.empty((0, 21, 3))), dtype=np.float32
            ),
            right_hand_landmarks=np.asarray(
                data.get("right_hand_landmarks", np.empty((0, 21, 3))), dtype=np.float32
            ),
            left_wrist_velocity=np.asarray(
                data.get("left_wrist_velocity", np.empty((0,), dtype=np.float32)), dtype=np.float32
            ),
            right_wrist_velocity=np.asarray(
                data.get("right_wrist_velocity", np.empty((0,), dtype=np.float32)), dtype=np.float32
            ),
        )

    @property
    def frames(self) -> int:
        return len(self.pose_state)

    def pose_at(self, frame_idx: int) -> str:
        if not self.pose_state:
            return "neutral_desk"
        frame_idx = max(0, min(frame_idx, len(self.pose_state) - 1))
        return self.pose_state[frame_idx]

    def active_segment_bounds(self) -> tuple[int, int] | None:
        if not self.transition_frames:
            return None
        return (self.transition_frames[0], self.transition_frames[-1])

    def iter_state_runs(self) -> Sequence[tuple[str, int, int]]:
        if not self.pose_state:
            return []
        runs: list[tuple[str, int, int]] = []
        start = 0
        current = self.pose_state[0]
        for idx, state in enumerate(self.pose_state[1:], start=1):
            if state != current:
                runs.append((current, start, idx))
                current = state
                start = idx
        runs.append((current, start, len(self.pose_state)))
        return runs
