from providers.motion_extraction.mediapipe.gesture_planner import RuleBasedGesturePlanner
from src.motion.composer import MotionTimelineComposer
from src.application.plan_video import VideoPlanner

class PlannerWorker:
    def __init__(self):
        self.planner = RuleBasedGesturePlanner()
        self.composer = MotionTimelineComposer()
        self.orchestrator = VideoPlanner(self.planner, self.composer)

    def process_job(self, avatar_id: str, text: str, language: str, voice_id: str, words_timestamps: list) -> dict:
        timeline = self.orchestrator.plan_video(avatar_id, text, language, voice_id, words_timestamps)
        return {
            "status": "planned",
            "timeline": timeline.model_dump()
        }
