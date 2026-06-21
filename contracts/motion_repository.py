from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pydantic import BaseModel

class MotionClip(BaseModel):
    root_translation: List[List[float]]
    root_rotation: List[List[float]]
    body_rotations: List[List[List[float]]]
    left_hand_rotations: List[List[List[float]]]
    right_hand_rotations: List[List[List[float]]]
    head_rotation: List[List[float]]
    confidence: List[List[float]]
    phase: List[str]

    class Config:
        arbitrary_types_allowed = True

class MotionRepository(ABC):
    @abstractmethod
    def get_motion(self, gesture_id: str) -> MotionClip:
        """Retrieve a motion clip from the repository."""
        pass

    @abstractmethod
    def save_motion(self, gesture_id: str, clip: MotionClip) -> None:
        """Save/version a motion clip in the repository."""
        pass
