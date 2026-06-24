"""Text-driven gesture timeline composer.

The canonical :class:`MotionTimelineComposer` in :mod:`src.motion.composer`
needs a Whisper-derived ``words_timestamps`` list to schedule gestures.
For "scripted demo" use cases — short videos, pre-edited scripts,
presenter-style one-liners — a full ASR round-trip is overkill.

This module plugs that hole:

1. Run the existing :class:`RuleBasedGesturePlanner` to get a small set of
   :class:`GestureIntent` rows from the text alone (Italian + English
   keywords already supported by the planner).
2. Distribute the intents across a caller-supplied ``audio_duration`` by
   weighting each anchor word's position in the original text. This is
   the character-offset heuristic the planner authors recommended
   historically — predictable, dependency-free, and good enough for
   vibe-style motion placement where there is no transcribed audio.
3. Fill the gaps between (and around) the intents with ``idle_small``
   segments so the timeline is contiguous.
4. Resolve each intent's duration from :class:`GestureRegistry` so the
   timeline durations come from the same YAML the production pipeline
   uses; no duplicated dictionaries.
5. Map each ``gesture_id`` to a canonical ``pose_id`` using the same
   table the rest of :mod:`src.motion` uses
   (:func:`pick_pose_id_for_gesture`).
6. Emit a :class:`Timeline` with the slimming-plan Change 4 shape so a
   follow-up :func:`align_timeline` pass can consume it without a
   separate adapter.

The module is import-safe on a stock Python install: it only uses
``pydantic`` (typed dataclasses) and ``pyyaml`` (registry reader) plus
the stdlib. There is no numpy, no cv2, no MediaPipe import — so the
unit tests + the demo CLI run on any CI host even if the GPU
inference path won't.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence

from contracts.gesture_planner import GestureIntent
from src.motion.registry import GestureRegistry


# ─────────────────────────────────────────────────────────────────────────────
# Canonical timeline shape (Change 4 of the slimming plan)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class TimelineSegment:
    """One segment in an avatar gesture timeline.

    Mirrors the slimming-plan Change 4 layout (see
    ``docs/REPOSITORY_SLIMMING_PLAN.md`` §10): a flat list of segments
    is enough to drive ``align_timeline`` against a body template
    library, no nested motion graph required.

    ``kind`` is either ``"idle"`` (rest pose) or ``"gesture"``
    (active stroke). ``pose_id`` is one of the keys in
    ``registry/hand_poses.yaml`` so the renderer can look up a body
    template resolution.
    """

    kind: str  # "idle" | "gesture"
    start: float
    end: float
    gesture_id: str
    pose_id: str
    intensity: float = 0.0
    anchor_word: str = ""
    text_span: str = ""


@dataclass(slots=True, frozen=True)
class Timeline:
    """Canonical Change 4 timeline.

    Compatible with :func:`src.pipeline.timeline_align.align_timeline`
    when it ships: round-trip through
    :meth:`Timeline.to_dict`/ :func:`Timeline.from_dict` is the contract.
    """

    duration: float
    fps: int
    segments: tuple[TimelineSegment, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration": self.duration,
            "fps": self.fps,
            "segments": [
                {
                    "kind": seg.kind,
                    "start": round(seg.start, 4),
                    "end": round(seg.end, 4),
                    "gesture_id": seg.gesture_id,
                    "pose_id": seg.pose_id,
                    "intensity": round(seg.intensity, 4),
                    "anchor_word": seg.anchor_word,
                    "text_span": seg.text_span,
                }
                for seg in self.segments
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Timeline":
        segments = tuple(
            TimelineSegment(
                kind=str(s["kind"]),
                start=float(s["start"]),
                end=float(s["end"]),
                gesture_id=str(s["gesture_id"]),
                pose_id=str(s["pose_id"]),
                intensity=float(s.get("intensity", 0.0)),
                anchor_word=str(s.get("anchor_word", "")),
                text_span=str(s.get("text_span", "")),
            )
            for s in payload.get("segments", [])
        )
        return cls(
            duration=float(payload["duration"]),
            fps=int(payload.get("fps", 25)),
            segments=segments,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pose mapping
# ─────────────────────────────────────────────────────────────────────────────


_GESTURE_TO_POSE: dict[str, str] = {
    "point_left": "left_hand_up",
    "explain_left": "left_hand_up",
    "point_right": "right_hand_up",
    "explain_right": "right_hand_up",
    "explain_both": "both_hands_open",
    "open_palms": "both_hands_open",
    "comparison": "both_hands_open",
}


def pick_pose_id_for_gesture(gesture_id: str) -> str:
    """Map a gesture_id from ``registry/gestures.yaml`` to a pose_id.

    The fallback pose is ``neutral_desk`` (the canonical rest pose).
    Same fallback logic as :func:`src.motion.composer._pick_pose_id`,
    duplicated here on purpose so this module does not import the
    composer (which still takes words_timestamps — circular if we want
    to plug one into the other).
    """
    return _GESTURE_TO_POSE.get(gesture_id, "neutral_desk")


# ─────────────────────────────────────────────────────────────────────────────
# Duration resolver
# ─────────────────────────────────────────────────────────────────────────────


class _GestureDurationLookup:
    """Read-only view of :class:`GestureRegistry` for timing lookups.

    Lives next to :func:`text_to_timeline` so the planner + composer
    can be swapped at the boundary without the duration resolver
    dragging in the pydantic BaseModel machinery.
    """

    def __init__(self, registry: GestureRegistry) -> None:
        self._registry = registry

    def duration_for(self, gesture_id: str, *, default: float = 3.0) -> float:
        if gesture_id not in self._registry.gestures:
            return default
        entry = self._registry.get_gesture(gesture_id)
        return float(entry.duration_seconds)


# ─────────────────────────────────────────────────────────────────────────────
# Text-driven composer
# ─────────────────────────────────────────────────────────────────────────────


def _normalise_text(text: str) -> str:
    """Lower-case + collapse whitespace so anchors match reliably."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _find_anchor_index(normalised_text: str, anchor_word: str) -> float:
    """Character-offset of ``anchor_word`` in ``normalised_text`` as a
    fraction in [0.0, 1.0]. Returns 0.0 for the empty text case so
    callers don't divide by zero.
    """
    if not normalised_text:
        return 0.0
    idx = normalised_text.find(anchor_word.lower().strip())
    if idx < 0:
        # Anchor not found: fall back to the geometric centre so the
        # gesture fires somewhere visible. Mirrors the planner's
        # behaviour of "schedule something even if no keyword matched".
        return 0.5
    # bias the start slightly earlier than the keyword so the gesture
    # peaks WITH the word (lead-in effect, defaults to 18-28% of the
    # segment duration — picked as the midpoint).
    return max(0.0, min(1.0, idx / max(1, len(normalised_text))))


def _plan_segments(
    intents: Sequence[GestureIntent],
    audio_duration: float,
    registry: _GestureDurationLookup,
    *,
    original_text: str = "",
) -> List[TimelineSegment]:
    """Convert intents to non-overlapping segments.

    The algorithm is fully deterministic:

    1. Sort intents by their anchor_offset (earlier in text → earlier
       in timeline).
    2. Reserve each intent's registered duration, anchored at
       ``audio_duration * anchor_offset``, with a small lead-in so
       the gesture peaks WITH the keyword rather than after.
    3. If two segments overlap, push the second ones until after the
       first one's end (preserves intent order without skipping).
    4. Fill the gap from 0 → first segment with idle, and between
       every consecutive gesture pair with idle too.
    5. The trailing gap gets a final idle segment so the schema
       ``segments are contiguous, ``segments[-1].end == duration``.

    ``original_text`` is the full script the user typed (preferred
    source of truth for anchor positions). When it is missing we
    fall back to joining the per-intent ``text_span`` strings, which
    biases offsets toward zero because the spans are short relative
    to the full text. Forwarding the full transcript makes the
    anchor positions land at their real on-screen position instead
    of clustering at t≈0.
    """
    if audio_duration <= 0.0:
        audio_duration = 0.0

    normalised = _normalise_text(original_text)
    if not normalised:
        # Fallback: join the per-intent text spans. The planner only
        # passes the anchor word as text_span for one-word triggers,
        # so this is short — but it preserves textual order and is
        # good enough when the full transcript isn't available.
        normalised = _normalise_text(
            " ".join(i.text_span or i.anchor_word or "" for i in intents)
        )

    # Compute proposed anchor time for each intent.
    proposed: list[tuple[float, GestureIntent, str]] = []
    for intent in intents:
        if not intent.anchor_word:
            # No anchor → bias by intent order (the planner already
            # returns intents in textual order).
            order_hint = len(proposed)
            proposed.append(
                (
                    (order_hint + 1) / max(1.0, len(intents) + 1.0),
                    intent,
                    intent.anchor_word or "",
                )
            )
            continue
        # Use the original text (not normalised intents) for richer
        # anchor-finding: the caller's transcript is denser than the
        # union of the planner's text spans.
        offset = _find_anchor_index(normalised, intent.anchor_word)
        proposed.append((offset, intent, intent.anchor_word))

    proposed.sort(key=lambda t: t[0])

    resolved: list[tuple[float, float, GestureIntent]] = []
    cursor = 0.0
    for offset, intent, _ in proposed:
        duration = registry.duration_for(intent.gesture_id, default=3.0)
        # Lead-in: gesture fires 18-28% BEFORE the keyword hit so the
        # stroke peaks with the word. The 22% midpoint is the default
        # recommended in the slim plan §6 timeline example.
        lead_in = max(0.0, duration * 0.22)
        start = max(cursor, audio_duration * offset - lead_in)
        end = start + duration
        if end > audio_duration:
            # The intent would overshoot the audio. Clip the END
            # back to audio_duration, and shift the START forward
            # (unconstrained only by ``cursor``, never below it) so
            # consecutive segments stay strictly non-overlapping.
            end = audio_duration
            start = max(cursor, end - duration)
        resolved.append((start, end, intent))
        cursor = end

    return _serialize(resolved, audio_duration)


def _serialize(
    resolved: Sequence[tuple[float, float, GestureIntent]],
    audio_duration: float,
) -> List[TimelineSegment]:
    """Insert idle segments so the timeline spans ``[0, audio_duration]``."""
    out: list[TimelineSegment] = []
    cursor = 0.0

    for start, end, intent in resolved:
        if start > cursor + 1e-3:
            out.append(
                TimelineSegment(
                    kind="idle",
                    start=cursor,
                    end=start,
                    gesture_id="idle_small",
                    pose_id="neutral_desk",
                    intensity=0.0,
                    anchor_word="",
                    text_span="",
                )
            )
        out.append(
            TimelineSegment(
                kind="gesture",
                start=start,
                end=end,
                gesture_id=intent.gesture_id,
                pose_id=pick_pose_id_for_gesture(intent.gesture_id),
                intensity=float(intent.intensity),
                anchor_word=intent.anchor_word,
                text_span=intent.text_span,
            )
        )
        cursor = end

    # Trailing idle so the timeline is contiguous.
    if audio_duration > cursor + 1e-3:
        out.append(
            TimelineSegment(
                kind="idle",
                start=cursor,
                end=audio_duration,
                gesture_id="idle_small",
                pose_id="neutral_desk",
                intensity=0.0,
                anchor_word="",
                text_span="",
            )
        )
    elif out and abs(out[-1].end - audio_duration) < 1e-3:
        out[-1] = TimelineSegment(
            kind=out[-1].kind,
            start=out[-1].start,
            end=audio_duration,
            gesture_id=out[-1].gesture_id,
            pose_id=out[-1].pose_id,
            intensity=out[-1].intensity,
            anchor_word=out[-1].anchor_word,
            text_span=out[-1].text_span,
        )

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def text_to_timeline(
    text: str,
    audio_duration: float,
    *,
    avatar_id: str = "default",
    language: str = "it",
    planner_factory: Optional[Callable[[], Any]] = None,
    registry_path: Path = Path("registry/gestures.yaml"),
    fps: int = 25,
) -> Timeline:
    """Build a :class:`Timeline` from raw text + an audio duration.

    Parameters
    ----------
    text:
        The script to be spoken. Keyword schedule leaks through the
        :class:`RuleBasedGesturePlanner` to produce ``GestureIntents``.
    audio_duration:
        Estimated clip length in seconds. Used to place intents on
        the wall clock and to size the trailing idle segment.
    avatar_id:
        Forwarded to the planner for per-avatar overrides. Currently
        unused by the rule-based planner.
    language:
        Forwarded to the planner for keyword pack selection. Currently
        unused by the rule-based planner.
    planner_factory:
        Optional callable returning a planner object (with a
        ``.plan(text, avatar_id, language)`` method). Defaults to
        :class:`RuleBasedGesturePlanner`.
    registry_path:
        YAML path :class:`GestureRegistry` reads. Defaults to the
        repo commit-tracked path.
    fps:
        Output framerate. Defaults to 25 (matches
        ``registry/models.yaml::standard.rationale``).
    """
    if planner_factory is None:
        from providers.motion_extraction.mediapipe.gesture_planner import (
            RuleBasedGesturePlanner,
        )

        planner = RuleBasedGesturePlanner()
    else:
        planner = planner_factory()

    intents: Iterable[GestureIntent] = planner.plan(text, avatar_id, language)
    intents_list = list(intents)
    registry = GestureRegistry(registry_file=registry_path)
    duration_lookup = _GestureDurationLookup(registry)

    if not intents_list or audio_duration <= 0.0:
        # Degenerate input → single idle segment that fills the
        # duration so the downstream renderer has something to align.
        segments = (
            TimelineSegment(
                kind="idle",
                start=0.0,
                end=max(0.0, audio_duration),
                gesture_id="idle_small",
                pose_id="neutral_desk",
                intensity=0.0,
            ),
        )
        return Timeline(duration=max(0.0, audio_duration), fps=fps, segments=segments)

    segments = _plan_segments(intents_list, audio_duration, duration_lookup, original_text=text)
    return Timeline(
        duration=audio_duration,
        fps=fps,
        segments=tuple(segments),
    )


__all__ = [
    "Timeline",
    "TimelineSegment",
    "pick_pose_id_for_gesture",
    "text_to_timeline",
]
