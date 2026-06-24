"""Lightweight hand-free facial motion timelines.

This module mirrors the body-gesture timeline flow, but keeps the
vocabulary intentionally tiny so it can be reused as a cheap default
layer for avatar presentation: idle, blink, brow raise, smile, nod,
and question.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

from src.motion.face_planner import FaceMotionIntent, RuleBasedFaceMotionPlanner
from src.motion.face_registry import FaceMotionRegistry


@dataclass(slots=True, frozen=True)
class FaceMotionSegment:
    kind: str
    start: float
    end: float
    motion_id: str
    intensity: float = 0.0
    anchor_word: str = ""
    text_span: str = ""
    family: str = "expression"


@dataclass(slots=True, frozen=True)
class FaceMotionTimeline:
    duration: float
    fps: int
    segments: tuple[FaceMotionSegment, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration": self.duration,
            "fps": self.fps,
            "segments": [
                {
                    "kind": seg.kind,
                    "start": round(seg.start, 4),
                    "end": round(seg.end, 4),
                    "motion_id": seg.motion_id,
                    "intensity": round(seg.intensity, 4),
                    "anchor_word": seg.anchor_word,
                    "text_span": seg.text_span,
                    "family": seg.family,
                }
                for seg in self.segments
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FaceMotionTimeline":
        segments = tuple(
            FaceMotionSegment(
                kind=str(item.get("kind", "gesture")),
                start=float(item["start"]),
                end=float(item["end"]),
                motion_id=str(item["motion_id"]),
                intensity=float(item.get("intensity", 0.0)),
                anchor_word=str(item.get("anchor_word", "")),
                text_span=str(item.get("text_span", "")),
                family=str(item.get("family", "expression")),
            )
            for item in payload.get("segments", [])
        )
        return cls(duration=float(payload["duration"]), fps=int(payload.get("fps", 25)), segments=segments)

    def slice(self, start: float, end: float) -> "FaceMotionTimeline":
        clipped: list[FaceMotionSegment] = []
        for seg in self.segments:
            seg_start = max(start, seg.start)
            seg_end = min(end, seg.end)
            if seg_end <= seg_start:
                continue
            clipped.append(
                FaceMotionSegment(
                    kind=seg.kind,
                    start=seg_start - start,
                    end=seg_end - start,
                    motion_id=seg.motion_id,
                    intensity=seg.intensity,
                    anchor_word=seg.anchor_word,
                    text_span=seg.text_span,
                    family=seg.family,
                )
            )
        return FaceMotionTimeline(duration=max(0.0, end - start), fps=self.fps, segments=tuple(clipped))

    def write_json(self, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return dest


_MOTION_TO_FAMILY = {
    "face_idle_soft": "idle",
    "blink_soft": "eye",
    "brow_raise_small": "brow",
    "smile_small": "mouth",
    "nod_small": "head",
    "question_face": "expression",
}


def _motion_duration(motion_id: str, registry: FaceMotionRegistry) -> float:
    if motion_id in registry.motions:
        return float(registry.get_motion(motion_id).duration_seconds)
    return 0.8


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _anchor_fraction(text: str, anchor_word: str, fallback_index: int, intent_count: int) -> float:
    normalised = _normalise_text(text)
    if not normalised:
        return fallback_index / max(1, intent_count)
    needle = anchor_word.lower().strip()
    if not needle:
        return fallback_index / max(1, intent_count)
    idx = normalised.find(needle)
    if idx < 0:
        return fallback_index / max(1, intent_count)
    return max(0.0, min(1.0, idx / max(1, len(normalised))))


def _plan_segments(
    intents: Sequence[FaceMotionIntent],
    audio_duration: float,
    registry: FaceMotionRegistry,
    *,
    original_text: str = "",
) -> List[FaceMotionSegment]:
    if audio_duration <= 0.0:
        return [
            FaceMotionSegment(
                kind="idle",
                start=0.0,
                end=0.0,
                motion_id="face_idle_soft",
                intensity=0.0,
                anchor_word="",
                text_span="",
                family="idle",
            )
        ]

    ordered = sorted(
        enumerate(intents),
        key=lambda pair: (
            _anchor_fraction(original_text, pair[1].anchor_word, pair[0], len(intents)),
            pair[0],
        ),
    )

    out: list[FaceMotionSegment] = []
    current = 0.0

    if not ordered:
        return [
            FaceMotionSegment(
                kind="idle",
                start=0.0,
                end=audio_duration,
                motion_id="face_idle_soft",
                intensity=0.0,
                anchor_word="",
                text_span="",
                family="idle",
            )
        ]

    for idx, intent in ordered:
        duration = _motion_duration(intent.motion_id, registry)
        anchor_fraction = _anchor_fraction(original_text, intent.anchor_word, idx, len(intents))
        anchor_time = audio_duration * anchor_fraction
        lead_in = max(0.10, min(0.45, duration * 0.25))
        start = max(current, anchor_time - lead_in)
        end = min(audio_duration, start + duration)
        if start > current + 1e-3:
            out.append(
                FaceMotionSegment(
                    kind="idle",
                    start=current,
                    end=start,
                    motion_id="face_idle_soft",
                    intensity=0.0,
                    anchor_word="",
                    text_span="",
                    family="idle",
                )
            )
        out.append(
            FaceMotionSegment(
                kind="gesture",
                start=start,
                end=end,
                motion_id=intent.motion_id,
                intensity=float(intent.intensity),
                anchor_word=intent.anchor_word,
                text_span=intent.text_span,
                family=_MOTION_TO_FAMILY.get(intent.motion_id, "expression"),
            )
        )
        current = end

    if current < audio_duration:
        out.append(
            FaceMotionSegment(
                kind="idle",
                start=current,
                end=audio_duration,
                motion_id="face_idle_soft",
                intensity=0.0,
                anchor_word="",
                text_span="",
                family="idle",
            )
        )
    return out


def text_to_face_motion_timeline(
    text: str,
    audio_duration: float,
    *,
    avatar_id: str = "default",
    language: str = "it",
    planner_factory: Optional[Callable[[], Any]] = None,
    registry_path: Path = Path("registry/facial_motions.yaml"),
    fps: int = 25,
) -> FaceMotionTimeline:
    if planner_factory is None:
        planner_factory = RuleBasedFaceMotionPlanner

    planner = planner_factory()
    registry = FaceMotionRegistry(registry_file=registry_path)
    intents = planner.plan(text, avatar_id, language)
    segments = _plan_segments(intents, audio_duration, registry, original_text=text)
    return FaceMotionTimeline(duration=float(audio_duration), fps=fps, segments=tuple(segments))

