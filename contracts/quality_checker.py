from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from pydantic import BaseModel

class QCRequest(BaseModel):
    video_path: Path
    audio_path: Path

class QCResult(BaseModel):
    passed: bool
    warnings: List[str]
    errors: List[str]
    status: str

class QualityChecker(ABC):
    @abstractmethod
    def check_quality(self, request: QCRequest) -> QCResult:
        """Run post-production validation checks on the final MP4."""
        pass
