from pathlib import Path
from typing import List, Dict, Any
from contracts.gesture_planner import GesturePlanner, GestureIntent
from src.motion.composer import MotionTimelineComposer, MotionTimeline

class VideoPlanner:
    def __init__(self, planner: GesturePlanner, composer: MotionTimelineComposer):
        self.planner = planner
        self.composer = composer

    def plan_video(
        self,
        avatar_id: str,
        text: str,
        language: str,
        voice_id: str,
        words_timestamps: List[Dict[str, Any]]
    ) -> MotionTimeline:
        intents = self.planner.plan(text, avatar_id, language)
        timeline = self.composer.compose(intents, words_timestamps)
        return timeline
