"""Compose speaking motion timelines from gesture intents and word timing."""

from __future__ import annotations

from typing import Any, List, Sequence

from pydantic import BaseModel

from contracts.gesture_planner import GestureIntent


class TimelineSegment(BaseModel):
    kind: str = "gesture"
    start: float
    end: float
    gesture_id: str = "idle_small"
    pose_id: str = "neutral_desk"
    stroke_time: float = 0.0
    anchor_word: str = ""
    intensity: float = 0.0


class MotionTimeline(BaseModel):
    duration: float
    segments: List[TimelineSegment]


def _gesture_duration(gesture_id: str) -> float:
    base = {
        "idle_small": 4.0,
        "idle_medium": 4.0,
        "open_palms": 3.0,
        "emphasis_small": 2.0,
        "emphasis_large": 3.0,
        "question": 2.5,
        "agreement": 2.0,
        "conclusion": 3.5,
        "call_to_action": 4.0,
    }
    return base.get(gesture_id, 3.0)


def _pick_pose_id(gesture_id: str) -> str:
    if gesture_id in {"point_left", "explain_left"}:
        return "left_hand_up"
    if gesture_id in {"point_right", "explain_right"}:
        return "right_hand_up"
    if gesture_id in {"explain_both", "open_palms", "comparison"}:
        return "both_hands_open"
    return "neutral_desk"


def _normalise_words(words_timestamps: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in words_timestamps:
        if not isinstance(item, dict):
            continue
        word = str(item.get("word", "")).strip().lower()
        if not word:
            continue
        normalized.append(
            {
                "word": word,
                "start": float(item.get("start", 0.0) or 0.0),
                "end": float(item.get("end", item.get("start", 0.0)) or 0.0),
            }
        )
    return normalized


class MotionTimelineComposer:
    def compose(self, intents: List[GestureIntent], words_timestamps: List[dict]) -> MotionTimeline:
        normalized_words = _normalise_words(words_timestamps)
        segments: list[TimelineSegment] = []
        current_time = 0.0

        for intent in intents:
            duration = _gesture_duration(intent.gesture_id)
            stroke_time = current_time
            for word_info in normalized_words:
                if word_info["word"] == intent.anchor_word.lower():
                    stroke_time = word_info["start"]
                    break

            lead_in = max(0.18, min(0.72, 0.28 * duration))
            gesture_start = max(current_time, stroke_time - lead_in)
            if gesture_start > current_time + 0.01:
                segments.append(
                    TimelineSegment(
                        kind="idle",
                        start=current_time,
                        end=gesture_start,
                        gesture_id="idle_small",
                        pose_id="neutral_desk",
                        stroke_time=current_time,
                        anchor_word="",
                        intensity=0.0,
                    )
                )

            gesture_end = gesture_start + duration
            segments.append(
                TimelineSegment(
                    kind="gesture",
                    start=gesture_start,
                    end=gesture_end,
                    gesture_id=intent.gesture_id,
                    pose_id=_pick_pose_id(intent.gesture_id),
                    stroke_time=stroke_time,
                    anchor_word=intent.anchor_word,
                    intensity=float(intent.intensity),
                )
            )
            current_time = gesture_end

        if not segments:
            segments.append(
                TimelineSegment(
                    kind="idle",
                    start=0.0,
                    end=2.0,
                    gesture_id="idle_small",
                    pose_id="neutral_desk",
                    stroke_time=0.0,
                    anchor_word="",
                    intensity=0.0,
                )
            )
            current_time = 2.0
        return MotionTimeline(duration=current_time, segments=segments)
