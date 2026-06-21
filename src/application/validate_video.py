from pathlib import Path
from contracts.quality_checker import QualityChecker, QCRequest, QCResult

class VideoValidator:
    def __init__(self, checker: QualityChecker):
        self.checker = checker

    def validate(self, video_path: Path, audio_path: Path) -> QCResult:
        request = QCRequest(video_path=video_path, audio_path=audio_path)
        return self.checker.check_quality(request)
