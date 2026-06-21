from typing import List
from pydantic import BaseModel
from contracts.gesture_planner import GestureIntent

class TimelineSegment(BaseModel):
    start: float
    end: float
    gesture_id: str
    stroke_time: float = 0.0

class MotionTimeline(BaseModel):
    duration: float
    segments: List[TimelineSegment]

class MotionTimelineComposer:
    def compose(self, intents: List[GestureIntent], words_timestamps: List[dict]) -> MotionTimeline:
        segments = []
        current_time = 0.0
        
        # Simple timeline builder aligning intents to word timestamps
        for intent in intents:
            # Find the word timestamp for anchor word
            stroke_time = current_time
            duration = 3.0  # baseline duration
            for word_info in words_timestamps:
                if word_info.get("word") == intent.anchor_word:
                    stroke_time = word_info.get("start", current_time)
                    break
            
            end_time = current_time + duration
            segments.append(TimelineSegment(
                start=current_time,
                end=end_time,
                gesture_id=intent.gesture_id,
                stroke_time=stroke_time
            ))
            current_time = end_time
            
        return MotionTimeline(
            duration=current_time,
            segments=segments
        )
