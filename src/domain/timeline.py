"""Timeline dataclass + JSON loader for deterministic multi-template rendering.

Per ``docs/REPOSITORY_SLIMMING_PLAN.md`` §6 + §10 (Change 4 of
the slim plan), the MVP ships a manual body-template timeline before
any LLM-style gesture planner is wired in. This dataclass lifts the
JSON shape into a frozen type so the orchestrator, the frame-align
utility, and the test suite all share the same canonical
representation.

Lifecycle
---------
* The CLI / API parses a timeline ``Path`` (or JSON dict) into a
  :class:`Timeline` via :func:`Timeline.from_json` /
  :func:`Timeline.from_dict`.
* The orchestrator iterates ``timeline.segments`` and calls
  :func:`src.pipeline.timeline_align.align_timeline` once per render
  — the align utility reads the timeline and writes ONE concatenated
  body / face_mask / neck_mask / face_transforms.npz on disk.
* The downstream compositor (``src.pipeline.compositor.OpenCVFaceCompositor``)
  consumes the aligned assets as if they came from a single
  :class:`body_template.BodyTemplate`.

Segment JSON shape
------------------
::

    {
        "gesture_id": "explain_both",
        "duration_seconds": 2.0
    }

Total duration is the sum of segment durations; the frame-align
utility enforces a strict frame-count invariant::

    round(segment.duration_seconds * timeline.fps)
    == body_template.frame_count()

(the source template must have been precomputed at exactly that
duration). A zero / negative duration raises at load time; a missing
key raises a structured :class:`ValueError` so the orchestrator can
fail fast at API time, well before the GPU worker is invoked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

# Matches `registry/models.yaml::standard.rationale` (25 FPS canonical).
DEFAULT_TIMELINE_FPS = 25


@dataclass(slots=True, frozen=True)
class TimelineSegment:
    """One segment of the manual timeline."""

    gesture_id: str
    duration_seconds: float


@dataclass(slots=True, frozen=True)
class Timeline:
    """An ordered chain of body-template segments + the canonical fps.

    ``fps`` is the single source of truth for the align utility: the
    duration-seconds × fps math is done once at the Timeline level
    so every per-segment frame count is consistent.
    """

    segments: Tuple[TimelineSegment, ...]
    fps: int = DEFAULT_TIMELINE_FPS

    # ── constructors ─────────────────────────────────────────────────────────

    @classmethod
    def from_json(cls, path: Path | str) -> "Timeline":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "Timeline":
        if not isinstance(data, dict) or "segments" not in data:
            raise ValueError(
                "Timeline JSON must be an object with a 'segments' list"
            )

        raw_segments = data.get("segments") or []
        if not isinstance(raw_segments, list):
            raise ValueError(
                f"Timeline 'segments' must be a list; got "
                f"{type(raw_segments).__name__}"
            )
        if len(raw_segments) == 0:
            raise ValueError("Timeline must contain at least one segment")

        segments: list[TimelineSegment] = []
        for idx, seg in enumerate(raw_segments):
            if not isinstance(seg, dict):
                raise ValueError(
                    f"Timeline.segments[{idx}] must be an object; "
                    f"got {type(seg).__name__}"
                )
            try:
                gesture_id = str(seg["gesture_id"])
                duration = float(seg["duration_seconds"])
            except KeyError as exc:
                raise ValueError(
                    f"Timeline.segments[{idx}] missing required key: "
                    f"{exc.args[0]!r}"
                ) from exc
            if duration <= 0:
                raise ValueError(
                    f"Timeline.segments[{idx}] gesture_id={gesture_id!r} "
                    f"has non-positive duration {duration}"
                )
            segments.append(
                TimelineSegment(gesture_id=gesture_id, duration_seconds=duration)
            )

        fps = int(data.get("fps", DEFAULT_TIMELINE_FPS))
        if fps <= 0:
            raise ValueError(f"Timeline fps must be positive; got {fps}")

        return cls(segments=tuple(segments), fps=fps)

    # ── predicates + derived properties ──────────────────────────────────────

    def is_well_formed(self) -> bool:
        return (
            len(self.segments) > 0
            and all(s.duration_seconds > 0 for s in self.segments)
            and all(s.gesture_id for s in self.segments)
            and self.fps > 0
        )

    def total_duration_seconds(self) -> float:
        return float(sum(s.duration_seconds for s in self.segments))

    def expected_frames(self) -> int:
        """Sum of per-segment frame counts (=global convention)."""
        return sum(self.frame_count_for_segment(i) for i in range(len(self.segments)))

    def frame_count_for_segment(self, idx: int) -> int:
        if not 0 <= idx < len(self.segments):
            raise IndexError(
                f"Timeline.frame_count_for_segment(idx={idx}) — "
                f"timeline only has {len(self.segments)} segments"
            )
        return int(round(self.segments[idx].duration_seconds * self.fps))

    # ── serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "fps": self.fps,
            "segments": [
                {
                    "gesture_id": s.gesture_id,
                    "duration_seconds": s.duration_seconds,
                }
                for s in self.segments
            ],
        }

    def to_json(self, path: Path | str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)


__all__ = ["DEFAULT_TIMELINE_FPS", "Timeline", "TimelineSegment"]
