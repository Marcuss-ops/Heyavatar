from pathlib import Path
from providers.compositing.ffmpeg.quality_checker import RuleBasedQualityChecker
from src.application.validate_video import VideoValidator

class QualityWorker:
    def __init__(self):
        self.checker = RuleBasedQualityChecker()
        self.validator = VideoValidator(self.checker)

    def process_qc(self, video_path: str, audio_path: str) -> dict:
        res = self.validator.validate(Path(video_path), Path(audio_path))
        return {
            "status": "qc_done",
            "passed": res.passed,
            "errors": res.errors,
            "warnings": res.warnings,
            "qc_status": res.status
        }
