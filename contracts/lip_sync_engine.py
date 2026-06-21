from abc import ABC, abstractmethod
from pathlib import Path
from pydantic import BaseModel

class LipSyncRequest(BaseModel):
    face_video_path: Path
    audio_path: Path

class LipSyncResult(BaseModel):
    lipsynced_face_video_path: Path

class LipSyncEngine(ABC):
    @abstractmethod
    def sync_lips(self, request: LipSyncRequest) -> LipSyncResult:
        """Apply audio-to-lip synchronization on the face crop."""
        pass
