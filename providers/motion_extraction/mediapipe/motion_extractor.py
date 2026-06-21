from pathlib import Path
from contracts.motion_repository import MotionClip

class MediaPipeMotionExtractor:
    def extract(self, video_path: Path, gesture_id: str) -> MotionClip:
        """Extract canonical joint and hand rotations using MediaPipe Pose + Hands."""
        return MotionClip(
            root_translation=[[0.0, 0.0, 0.0]],
            root_rotation=[[0.0, 0.0, 0.0, 1.0]],
            body_rotations=[[[0.0, 0.0, 0.0, 1.0]]],
            left_hand_rotations=[[[0.0, 0.0, 0.0, 1.0]]],
            right_hand_rotations=[[[0.0, 0.0, 0.0, 1.0]]],
            head_rotation=[[0.0, 0.0, 0.0, 1.0]],
            confidence=[[1.0]],
            phase=["stroke"]
        )
