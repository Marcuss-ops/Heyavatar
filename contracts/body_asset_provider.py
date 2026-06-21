from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple
from pydantic import BaseModel

class BodyAsset(BaseModel):
    body_video_path: Path
    face_mask_video_path: Path
    neck_mask_video_path: Path
    face_transforms_path: Path
    metadata_path: Path

class BodyAssetProvider(ABC):
    @abstractmethod
    def resolve_body_asset(
        self,
        avatar_id: str,
        gesture_id: str,
        outfit_id: str,
        camera_id: str,
        lighting_id: str,
        resolution: Tuple[int, int],
        fps: int,
    ) -> BodyAsset:
        """Resolve a body template clip from the cache or trigger generation."""
        pass
