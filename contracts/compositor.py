from abc import ABC, abstractmethod
from pathlib import Path
from pydantic import BaseModel

class CompositingRequest(BaseModel):
    body_video_path: Path
    lipsynced_face_video_path: Path
    face_mask_video_path: Path
    neck_mask_video_path: Path
    face_transforms_path: Path
    color_profile_path: Path

class CompositingResult(BaseModel):
    composited_video_path: Path

class Compositor(ABC):
    @abstractmethod
    def composite(self, request: CompositingRequest) -> CompositingResult:
        """Composite the lip-synced face ROI back onto the body timeline."""
        pass
