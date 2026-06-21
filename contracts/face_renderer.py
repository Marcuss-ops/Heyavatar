from abc import ABC, abstractmethod
from pathlib import Path
from pydantic import BaseModel

class FaceRenderRequest(BaseModel):
    avatar_id: str
    head_motion_timeline_path: Path
    fps: int
    face_region_only: bool

class FaceRenderResult(BaseModel):
    face_track_video_path: Path
    face_alpha_video_path: Path

class FaceRenderer(ABC):
    @abstractmethod
    def render_face(self, request: FaceRenderRequest) -> FaceRenderResult:
        """Render the isolated face ROI (e.g. 256x256) based on head motion."""
        pass
