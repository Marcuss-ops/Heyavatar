"""Unit tests for ``src.domain.timeline``.

Covers:

* :class:`Timeline` JSON round-trip (``from_json`` / ``to_json`` /
  ``from_dict``).
* Derived property helpers (``is_well_formed``,
  ``total_duration_seconds``, ``expected_frames``,
  ``frame_count_for_segment``).
* Validation errors on malformed JSON (empty segments, negative
  duration, missing keys, non-positive fps, type mismatch).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.domain.timeline import (
    DEFAULT_TIMELINE_FPS,
    Timeline,
    TimelineSegment,
)


# ─────────────────────────────────────────────────────────────────────────────
# Slim-plan canonical timeline JSON.
# Per docs/REPOSITORY_SLIMMING_PLAN.md §6.
# ─────────────────────────────────────────────────────────────────────────────

EXAMPLE_DICT = {
    "fps": 25,
    "segments": [
        {"gesture_id": "idle", "duration_seconds": 3.0},
        {"gesture_id": "explain_both", "duration_seconds": 2.0},
        {"gesture_id": "idle", "duration_seconds": 3.0},
    ],
}
# Default fps = 25 → 75+50+75 = 200 frames; total 8.0s.
EXPECTED_TOTAL_SECONDS = 8.0
EXPECTED_FRAMES_AT_25 = 200
EXPECTED_FRAMES_PER_SEGMENT = [75, 50, 75]


# ─────────────────────────────────────────────────────────────────────────────
# from_json / from_dict / to_json
# ─────────────────────────────────────────────────────────────────────────────


def test_from_dict_round_trips_slim_plan_example():
    timeline = Timeline.from_dict(EXAMPLE_DICT)

    assert timeline.fps == DEFAULT_TIMELINE_FPS
    assert len(timeline.segments) == 3
    assert timeline.segments[0] == TimelineSegment(
        gesture_id="idle", duration_seconds=3.0
    )
    assert timeline.segments[1].gesture_id == "explain_both"
    assert timeline.segments[2].duration_seconds == 3.0


def test_from_dict_round_trips_to_dict():
    timeline = Timeline.from_dict(EXAMPLE_DICT)
    assert timeline.to_dict() == EXAMPLE_DICT


def test_from_dict_round_trips_to_dict_with_omitted_default_fps():
    """When JSON omits fps, to_dict() defaults to 25; to_dict() always emits fps."""
    data_no_fps = {
        "segments": [
            {"gesture_id": "idle", "duration_seconds": 3.0},
        ]
    }
    timeline = Timeline.from_dict(data_no_fps)
    assert timeline.fps == 25
    assert timeline.to_dict() == {"fps": 25, "segments": data_no_fps["segments"]}


def test_from_json_file_round_trip(tmp_path: Path):
    src = tmp_path / "timeline.json"
    src.write_text(json.dumps(EXAMPLE_DICT), encoding="utf-8")

    timeline = Timeline.from_json(src)
    assert timeline.total_duration_seconds() == pytest.approx(
        EXPECTED_TOTAL_SECONDS
    )
    assert timeline.expected_frames() == EXPECTED_FRAMES_AT_25


def test_to_json_file_round_trip(tmp_path: Path):
    timeline = Timeline.from_dict(EXAMPLE_DICT)
    dest = tmp_path / "out.json"
    timeline.to_json(dest)

    reloaded = Timeline.from_json(dest)
    assert reloaded.fps == timeline.fps
    assert reloaded.segments == timeline.segments


def test_custom_fps_is_preserved():
    data = dict(EXAMPLE_DICT, fps=30)
    timeline = Timeline.from_dict(data)
    assert timeline.fps == 30
    # 8 seconds @ 30 fps = 240 frames.
    assert timeline.expected_frames() == 240
    assert timeline.frame_count_for_segment(0) == round(3 * 30)  # 90


# ─────────────────────────────────────────────────────────────────────────────
# Derived properties
# ─────────────────────────────────────────────────────────────────────────────


def test_total_duration_seconds_sums_segments():
    timeline = Timeline.from_dict(EXAMPLE_DICT)
    assert timeline.total_duration_seconds() == pytest.approx(
        EXPECTED_TOTAL_SECONDS
    )


def test_expected_frames_at_default_fps():
    timeline = Timeline.from_dict(EXAMPLE_DICT)
    assert timeline.expected_frames() == EXPECTED_FRAMES_AT_25


def test_frame_count_for_segment_per_segment():
    timeline = Timeline.from_dict(EXAMPLE_DICT)
    for idx, expected in enumerate(EXPECTED_FRAMES_PER_SEGMENT):
        assert timeline.frame_count_for_segment(idx) == expected, (
            f"segment {idx} mismatch"
        )


def test_frame_count_for_segment_out_of_bounds():
    timeline = Timeline.from_dict(EXAMPLE_DICT)
    with pytest.raises(IndexError):
        timeline.frame_count_for_segment(99)


def test_is_well_formed_true_for_valid_example():
    assert Timeline.from_dict(EXAMPLE_DICT).is_well_formed() is True


def test_is_well_formed_false_for_empty_segments():
    with pytest.raises(ValueError):
        Timeline.from_dict({"segments": []})


def test_is_well_formed_false_for_zero_duration():
    bad = {
        "segments": [
            {"gesture_id": "idle", "duration_seconds": 0.0},
        ]
    }
    with pytest.raises(ValueError) as exc:
        Timeline.from_dict(bad)
    assert "non-positive" in str(exc.value)


def test_is_well_formed_false_for_negative_duration():
    bad = {
        "segments": [
            {"gesture_id": "idle", "duration_seconds": -1.0},
        ]
    }
    with pytest.raises(ValueError) as exc:
        Timeline.from_dict(bad)
    assert "non-positive" in str(exc.value)


# ─────────────────────────────────────────────────────────────────────────────
# Validation: shape, types, missing keys
# ─────────────────────────────────────────────────────────────────────────────


def test_from_dict_rejects_non_object_root():
    with pytest.raises(ValueError):
        Timeline.from_dict([])  # type: ignore[arg-type]


def test_from_dict_rejects_missing_segments_key():
    with pytest.raises(ValueError):
        Timeline.from_dict({"fps": 25})


def test_from_dict_rejects_non_list_segments():
    with pytest.raises(ValueError) as exc:
        Timeline.from_dict({"segments": "not-a-list"})
    assert "list" in str(exc.value)


def test_from_dict_rejects_non_object_segment():
    with pytest.raises(ValueError) as exc:
        Timeline.from_dict({"segments": ["plain-string"]})
    assert "object" in str(exc.value)


def test_from_dict_rejects_segment_missing_gesture_id():
    bad = {"segments": [{"duration_seconds": 1.0}]}
    with pytest.raises(ValueError) as exc:
        Timeline.from_dict(bad)
    assert "gesture_id" in str(exc.value)


def test_from_dict_rejects_segment_missing_duration():
    bad = {"segments": [{"gesture_id": "idle"}]}
    with pytest.raises(ValueError) as exc:
        Timeline.from_dict(bad)
    assert "duration_seconds" in str(exc.value)


def test_from_dict_rejects_non_positive_fps():
    bad = dict(EXAMPLE_DICT, fps=0)
    with pytest.raises(ValueError) as exc:
        Timeline.from_dict(bad)
    assert "fps" in str(exc.value)


def test_from_dict_rejects_negative_fps():
    bad = dict(EXAMPLE_DICT, fps=-5)
    with pytest.raises(ValueError) as exc:
        Timeline.from_dict(bad)
    assert "fps" in str(exc.value)


# ─────────────────────────────────────────────────────────────────────────────
# Type semantics: gesture_id coerces to str, duration coerces to float
# ─────────────────────────────────────────────────────────────────────────────


def test_from_dict_coerces_string_gesture_id():
    data = {"segments": [{"gesture_id": 12345, "duration_seconds": 1.0}]}
    timeline = Timeline.from_dict(data)
    assert timeline.segments[0].gesture_id == "12345"
    assert isinstance(timeline.segments[0].gesture_id, str)


def test_from_dict_coerces_numeric_duration():
    data = {"segments": [{"gesture_id": "idle", "duration_seconds": 2}]}
    timeline = Timeline.from_dict(data)
    assert timeline.segments[0].duration_seconds == 2.0
    assert isinstance(timeline.segments[0].duration_seconds, float)


# ─────────────────────────────────────────────────────────────────────────────
# Frozen dataclass guarantees
# ─────────────────────────────────────────────────────────────────────────────


def test_timeline_is_frozen():
    timeline = Timeline.from_dict(EXAMPLE_DICT)
    with pytest.raises((AttributeError, Exception)):
        timeline.fps = 30  # type: ignore[misc]


def test_segment_is_frozen():
    segment = TimelineSegment(gesture_id="idle", duration_seconds=3.0)
    with pytest.raises((AttributeError, Exception)):
        segment.duration_seconds = 99.0  # type: ignore[misc]
