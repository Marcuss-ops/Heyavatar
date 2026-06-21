"""Unit tests for ``src.pipeline.timeline_align.align_timeline``.

Three-segment alignment is the canonical Change 4 acceptance case:

* 75-frame template at 25 fps for ``idle`` (segment 0).
* 50-frame template at 25 fps for ``explain_both`` (segment 1).
* 75-frame template at 25 fps for ``idle`` (segment 2).

The expected aligned product is 200 frames @ 25 fps = 8.0 s.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.domain.timeline import Timeline
from src.pipeline.timeline_align import AlignedBodyTimeline, align_timeline


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic body-template writer (test-local helper).
# Mirrors the layout of ``tools/avatar_assets/precompute_video_template.py``.
# ─────────────────────────────────────────────────────────────────────────────


def _synth_template(
    base_dir: Path,
    avatar_id: str,
    gesture_id: str,
    *,
    width: int,
    height: int,
    fps: float,
    total_frames: int,
    body_colour: tuple[int, int, int] = (128, 128, 128),
    fmask_colour: tuple[int, int, int] = (200, 200, 200),
    nmask_colour: tuple[int, int, int] = (80, 80, 80),
    bbox_x_pad: int = 8,
) -> Path:
    tmpl_dir = base_dir / avatar_id / gesture_id
    tmpl_dir.mkdir(parents=True, exist_ok=True)
    for name, colour in (
        ("body.mp4", body_colour),
        ("face_mask.mp4", fmask_colour),
        ("neck_mask.mp4", nmask_colour),
    ):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(tmpl_dir / name), fourcc, fps, (width, height))
        assert writer.isOpened(), f"failed to write {tmpl_dir / name}"
        frame = np.full((height, width, 3), colour, dtype=np.uint8)
        for _ in range(total_frames):
            writer.write(frame)
        writer.release()

    bbox = np.tile(
        np.array(
            [bbox_x_pad, bbox_x_pad, width - bbox_x_pad, height - bbox_x_pad],
            dtype=np.float32,
        ),
        (total_frames, 1),
    )
    matrices = np.tile(np.eye(4, dtype=np.float32), (total_frames, 1, 1))
    np.savez_compressed(
        tmpl_dir / "face_transforms.npz",
        bbox=bbox,
        matrices=matrices,
    )

    payload = {
        "avatar_id": avatar_id,
        "gesture_id": gesture_id,
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
        "status": "precomputed",
    }
    with open(tmpl_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return tmpl_dir


def _build_three_template_block(tmp_path: Path) -> Path:
    base = tmp_path / "body_templates"
    base.mkdir(parents=True, exist_ok=True)
    _synth_template(
        base, "alice", "idle", width=64, height=64,
        fps=25.0, total_frames=75,
        body_colour=(110, 110, 110), fmask_colour=(220, 0, 0),
    )
    _synth_template(
        base, "alice", "explain_both", width=64, height=64,
        fps=25.0, total_frames=50,
        body_colour=(90, 90, 140), fmask_colour=(0, 220, 0),
    )
    _synth_template(
        base, "alice", "idle", width=64, height=64,
        fps=25.0, total_frames=75,
        body_colour=(140, 110, 110), fmask_colour=(0, 0, 220),
    )
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Custom loader that points the align utility at our synthetic tree.
# ─────────────────────────────────────────────────────────────────────────────


def _local_loader(avatar_id: str, gesture_id: str) -> "BodyTemplate":
    """Stand-in for load_body_template that trusts synthetic files exist."""
    from src.domain.body_template import BodyTemplate

    base = _last_synthetic_base  # type: ignore[name-defined]
    return BodyTemplate(
        body_video=base / avatar_id / gesture_id / "body.mp4",
        face_mask=base / avatar_id / gesture_id / "face_mask.mp4",
        neck_mask=base / avatar_id / gesture_id / "neck_mask.mp4",
        face_transforms=base / avatar_id / gesture_id / "face_transforms.npz",
        metadata=base / avatar_id / gesture_id / "metadata.json",
    )


# A module-global mutable cache so the loader closure can find the
# synthetic body_templates root for the duration of one test. The
# fixture pattern below writes/restores this around the call.
_last_synthetic_base: Path = Path("/tmp/never")


@pytest.fixture
def synth_templates(tmp_path: Path):
    """Build a 3-template block + return a loader that finds them.

    The loader is monkey-closed over the tmp_path inside the test.
    """
    global _last_synthetic_base
    base = _build_three_template_block(tmp_path)
    previous = _last_synthetic_base
    _last_synthetic_base = base
    yield base, _local_loader
    _last_synthetic_base = previous


# ─────────────────────────────────────────────────────────────────────────────
# 3-segment acceptance case
# ─────────────────────────────────────────────────────────────────────────────


def test_align_three_segments_total_frames(synth_templates, tmp_path):
    base, loader = synth_templates
    timeline = Timeline.from_dict({
        "segments": [
            {"gesture_id": "idle", "duration_seconds": 3.0},
            {"gesture_id": "explain_both", "duration_seconds": 2.0},
            {"gesture_id": "idle", "duration_seconds": 3.0},
        ]
    })

    aligned = align_timeline(
        timeline, "alice", output_dir=tmp_path / "aligned",
        body_template_loader=loader,
    )

    assert aligned.total_frames == 200
    assert aligned.duration_seconds == pytest.approx(8.0)
    assert aligned.fps == 25
    assert aligned.width == 64
    assert aligned.height == 64


def test_align_writes_all_five_files(synth_templates, tmp_path):
    base, loader = synth_templates
    timeline = Timeline.from_dict({
        "segments": [
            {"gesture_id": "idle", "duration_seconds": 3.0},
            {"gesture_id": "explain_both", "duration_seconds": 2.0},
            {"gesture_id": "idle", "duration_seconds": 3.0},
        ]
    })

    output_dir = tmp_path / "aligned"
    aligned = align_timeline(
        timeline, "alice", output_dir=output_dir,
        body_template_loader=loader,
    )

    for path in (
        aligned.body_video, aligned.face_mask, aligned.neck_mask,
        aligned.face_transforms, aligned.metadata,
    ):
        assert path.is_file(), f"expected {path} to exist"


def test_align_npz_bbox_array_has_full_length(synth_templates, tmp_path):
    base, loader = synth_templates
    timeline = Timeline.from_dict({
        "segments": [
            {"gesture_id": "idle", "duration_seconds": 3.0},
            {"gesture_id": "explain_both", "duration_seconds": 2.0},
            {"gesture_id": "idle", "duration_seconds": 3.0},
        ]
    })
    aligned = align_timeline(
        timeline, "alice", output_dir=tmp_path / "aligned",
        body_template_loader=loader,
    )

    data = np.load(str(aligned.face_transforms))
    assert data["bbox"].shape == (200, 4)
    assert data["matrices"].shape == (200, 4, 4)
    assert data["timestamp_ms"].shape == (200,)


def test_align_npz_timestamps_are_strictly_monotonic(synth_templates, tmp_path):
    base, loader = synth_templates
    timeline = Timeline.from_dict({
        "segments": [
            {"gesture_id": "idle", "duration_seconds": 3.0},
            {"gesture_id": "explain_both", "duration_seconds": 2.0},
            {"gesture_id": "idle", "duration_seconds": 3.0},
        ]
    })
    aligned = align_timeline(
        timeline, "alice", output_dir=tmp_path / "aligned",
        body_template_loader=loader,
    )

    ts = np.load(str(aligned.face_transforms))["timestamp_ms"]
    # Strictly increasing.
    assert np.all(np.diff(ts) > 0)
    # Step is exactly 1000/fps = 40ms.
    assert np.all(np.diff(ts) == 40)


def test_align_timeline_break_boundary_at_segment_index_75(synth_templates, tmp_path):
    base, loader = synth_templates
    timeline = Timeline.from_dict({
        "segments": [
            {"gesture_id": "idle", "duration_seconds": 3.0},
            {"gesture_id": "explain_both", "duration_seconds": 2.0},
            {"gesture_id": "idle", "duration_seconds": 3.0},
        ]
    })
    aligned = align_timeline(
        timeline, "alice", output_dir=tmp_path / "aligned",
        body_template_loader=loader,
    )

    # Frame 0..74 = segment 0 (idle), 75..124 = segment 1 (explain_both),
    # 125..199 = segment 2 (idle again).
    ts = np.load(str(aligned.face_transforms))["timestamp_ms"]
    assert ts[0] == 0
    assert ts[74] == 74 * 40
    assert ts[75] == 75 * 40  # explicitly continues (no gap)
    assert ts[124] == 124 * 40
    assert ts[199] == 199 * 40


def test_align_custom_fps_preserved_through_output_metadata(
    synth_templates, tmp_path
):
    _, loader = synth_templates
    body_templates_base = tmp_path / "body_templates_30fps"
    body_templates_base.mkdir(parents=True, exist_ok=True)
    _synth_template(
        body_templates_base, "alice", "idle", width=64, height=64,
        fps=30.0, total_frames=90, body_colour=(120, 120, 120),
        fmask_colour=(220, 0, 0),
    )
    _synth_template(
        body_templates_base, "alice", "explain_both", width=64, height=64,
        fps=30.0, total_frames=60, body_colour=(90, 90, 140),
        fmask_colour=(0, 220, 0),
    )
    _synth_template(
        body_templates_base, "alice", "idle", width=64, height=64,
        fps=30.0, total_frames=90, body_colour=(140, 110, 110),
        fmask_colour=(0, 0, 220),
    )

    def loader_30(avatar_id, gesture_id):
        from src.domain.body_template import BodyTemplate
        return BodyTemplate(
            body_video=body_templates_base / avatar_id / gesture_id / "body.mp4",
            face_mask=body_templates_base / avatar_id / gesture_id / "face_mask.mp4",
            neck_mask=body_templates_base / avatar_id / gesture_id / "neck_mask.mp4",
            face_transforms=body_templates_base / avatar_id / gesture_id / "face_transforms.npz",
            metadata=body_templates_base / avatar_id / gesture_id / "metadata.json",
        )

    timeline = Timeline.from_dict({
        "fps": 30,
        "segments": [
            {"gesture_id": "idle", "duration_seconds": 3.0},
            {"gesture_id": "explain_both", "duration_seconds": 2.0},
            {"gesture_id": "idle", "duration_seconds": 3.0},
        ]
    })
    aligned = align_timeline(
        timeline, "alice", output_dir=tmp_path / "aligned_30",
        body_template_loader=loader_30,
    )

    assert aligned.fps == 30
    assert aligned.total_frames == 240
    assert aligned.duration_seconds == pytest.approx(8.0)
    # step in npz = 1000/30 ≈ 33ms.
    ts = np.load(str(aligned.face_transforms))["timestamp_ms"]
    assert np.all(np.diff(ts) == 33)


# ─────────────────────────────────────────────────────────────────────────────
# Failure modes — strict invariants
# ─────────────────────────────────────────────────────────────────────────────


def test_align_frame_count_mismatch_raises(synth_templates, tmp_path):
    base, loader = synth_templates

    # Replace the 75-frame idle template under body_templates/ with
    # a 70-frame stub so the per-segment frame-count invariant fires
    # on segment 0 (timeline prescribes 75 for 3.0s @ 25fps).
    _synth_template(
        base, "alice", "idle", width=64, height=64,
        fps=25.0, total_frames=70, fmask_colour=(220, 0, 0),
    )

    timeline = Timeline.from_dict({
        "segments": [
            {"gesture_id": "idle", "duration_seconds": 3.0},
            {"gesture_id": "explain_both", "duration_seconds": 2.0},
            {"gesture_id": "idle", "duration_seconds": 3.0},
        ]
    })

    with pytest.raises(ValueError) as exc:
        align_timeline(
            timeline, "alice", output_dir=tmp_path / "aligned",
            body_template_loader=loader,
        )
    msg = str(exc.value)
    assert "frame-count mismatch" in msg
    assert "70" in msg and "75" in msg


def test_align_dimension_mismatch_raises(tmp_path):
    base = tmp_path / "body_templates"
    base.mkdir(parents=True, exist_ok=True)
    # Two templates with the SAME frame count (prescribed fps * duration)
    # but DIFFERENT resolutions so the cross-segment dimension invariant fires.
    _synth_template(base, "alice", "idle", width=64, height=64,
                    fps=25.0, total_frames=75)
    _synth_template(base, "alice", "wide_gesture", width=80, height=80,
                    fps=25.0, total_frames=50)

    def loader(avatar_id, gesture_id):
        from src.domain.body_template import BodyTemplate
        return BodyTemplate(
            body_video=base / avatar_id / gesture_id / "body.mp4",
            face_mask=base / avatar_id / gesture_id / "face_mask.mp4",
            neck_mask=base / avatar_id / gesture_id / "neck_mask.mp4",
            face_transforms=base / avatar_id / gesture_id / "face_transforms.npz",
            metadata=base / avatar_id / gesture_id / "metadata.json",
        )

    timeline = Timeline.from_dict({
        "segments": [
            {"gesture_id": "idle", "duration_seconds": 3.0},
            {"gesture_id": "wide_gesture", "duration_seconds": 2.0},
        ]
    })

    with pytest.raises(ValueError) as exc:
        align_timeline(
            timeline, "alice", output_dir=tmp_path / "aligned",
            body_template_loader=loader,
        )
    assert "dimension mismatch" in str(exc.value)


def test_align_missing_face_transforms_raises(tmp_path):
    base = tmp_path / "body_templates"
    base.mkdir(parents=True, exist_ok=True)
    _synth_template(base, "alice", "idle", width=64, height=64,
                    fps=25.0, total_frames=75)
    _synth_template(base, "alice", "explain_both", width=64, height=64,
                    fps=25.0, total_frames=50)
    _synth_template(base, "alice", "alt_idle", width=64, height=64,
                    fps=25.0, total_frames=75)
    # Now overwrite the npz in the first segment to remove 'bbox'.
    import numpy as np
    np.savez_compressed(
        base / "alice" / "idle" / "face_transforms.npz",
        matrices=np.tile(np.eye(4, dtype=np.float32), (75, 1, 1)),
    )

    def loader(avatar_id, gesture_id):
        from src.domain.body_template import BodyTemplate
        return BodyTemplate(
            body_video=base / avatar_id / gesture_id / "body.mp4",
            face_mask=base / avatar_id / gesture_id / "face_mask.mp4",
            neck_mask=base / avatar_id / gesture_id / "neck_mask.mp4",
            face_transforms=base / avatar_id / gesture_id / "face_transforms.npz",
            metadata=base / avatar_id / gesture_id / "metadata.json",
        )

    timeline = Timeline.from_dict({
        "segments": [
            {"gesture_id": "idle", "duration_seconds": 3.0},
            {"gesture_id": "explain_both", "duration_seconds": 2.0},
            {"gesture_id": "alt_idle", "duration_seconds": 3.0},
        ]
    })

    with pytest.raises(ValueError) as exc:
        align_timeline(
            timeline, "alice", output_dir=tmp_path / "aligned",
            body_template_loader=loader,
        )
    assert "bbox" in str(exc.value)


def test_align_output_metadata_records_per_segment_breakdown(
    synth_templates, tmp_path
):
    _, loader = synth_templates
    timeline = Timeline.from_dict({
        "segments": [
            {"gesture_id": "idle", "duration_seconds": 3.0},
            {"gesture_id": "explain_both", "duration_seconds": 2.0},
            {"gesture_id": "idle", "duration_seconds": 3.0},
        ]
    })
    aligned = align_timeline(
        timeline, "alice", output_dir=tmp_path / "aligned",
        body_template_loader=loader,
    )

    with open(aligned.metadata, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["avatar_id"] == "alice"
    assert meta["fps"] == 25
    assert meta["width"] == 64
    assert meta["height"] == 64
    assert meta["total_frames"] == 200
    assert meta["duration_seconds"] == pytest.approx(8.0)
    assert meta["timeline_total_duration_seconds"] == pytest.approx(8.0)
    assert meta["status"] == "aligned"
    assert len(meta["segments"]) == 3
    assert [s["frames"] for s in meta["segments"]] == [75, 50, 75]


def test_align_aligned_to_body_template_shell(synth_templates, tmp_path):
    _, loader = synth_templates
    timeline = Timeline.from_dict({
        "segments": [
            {"gesture_id": "idle", "duration_seconds": 3.0},
            {"gesture_id": "explain_both", "duration_seconds": 2.0},
            {"gesture_id": "idle", "duration_seconds": 3.0},
        ]
    })
    aligned = align_timeline(
        timeline, "alice", output_dir=tmp_path / "aligned",
        body_template_loader=loader,
    )

    shell = aligned.as_body_template()
    assert shell.body_video == aligned.body_video
    assert shell.face_mask == aligned.face_mask
    assert shell.neck_mask == aligned.neck_mask
    assert shell.face_transforms == aligned.face_transforms
    assert shell.metadata == aligned.metadata


# ────────────────────────────────────────────────────────────────────────────
# Content-level invariants — MED 1 + MED 2 + LOW 5 follow-ups
# ────────────────────────────────────────────────────────────────────────────


def _corrupt_npz(template_dir: Path, *, bbox, matrices) -> None:
    """Overwrite the (already-written) face_transforms.npz in ``template_dir``
    with the provided bbox / matrices payloads so we can exercise the
    content-level invariants on a per-template basis.
    """
    np.savez_compressed(
        template_dir / "face_transforms.npz",
        bbox=bbox,
        matrices=matrices,
    )


def test_align_nan_bbox_raises(tmp_path):
    base = tmp_path / "body_templates"
    base.mkdir(parents=True, exist_ok=True)
    template_dir = _synth_template(
        base, "alice", "idle", width=64, height=64,
        fps=25.0, total_frames=75,
    )
    # Inject a NaN bbox row to trip the np.isfinite check.
    corrupt_bbox = np.tile(
        np.array([8.0, 8.0, 56.0, 56.0], dtype=np.float32), (75, 1)
    )
    corrupt_bbox[10] = [np.nan, 8.0, 56.0, 56.0]
    _corrupt_npz(
        template_dir,
        bbox=corrupt_bbox,
        matrices=np.tile(np.eye(4, dtype=np.float32), (75, 1, 1)),
    )

    def loader(avatar_id, gesture_id):
        from src.domain.body_template import BodyTemplate
        return BodyTemplate(
            body_video=base / avatar_id / gesture_id / "body.mp4",
            face_mask=base / avatar_id / gesture_id / "face_mask.mp4",
            neck_mask=base / avatar_id / gesture_id / "neck_mask.mp4",
            face_transforms=base / avatar_id / gesture_id / "face_transforms.npz",
            metadata=base / avatar_id / gesture_id / "metadata.json",
        )

    timeline = Timeline.from_dict({
        "segments": [{"gesture_id": "idle", "duration_seconds": 3.0}]
    })
    with pytest.raises(ValueError) as exc:
        align_timeline(
            timeline, "alice", output_dir=tmp_path / "aligned",
            body_template_loader=loader,
        )
    assert "non-finite" in str(exc.value)
    assert "bbox" in str(exc.value)


def test_align_out_of_bounds_bbox_raises(tmp_path):
    base = tmp_path / "body_templates"
    base.mkdir(parents=True, exist_ok=True)
    template_dir = _synth_template(
        base, "alice", "idle", width=64, height=64,
        fps=25.0, total_frames=75,
    )
    # x_max = 200, way beyond width=64.
    corrupt_bbox = np.tile(
        np.array([8.0, 8.0, 200.0, 56.0], dtype=np.float32), (75, 1)
    )
    _corrupt_npz(
        template_dir,
        bbox=corrupt_bbox,
        matrices=np.tile(np.eye(4, dtype=np.float32), (75, 1, 1)),
    )

    def loader(avatar_id, gesture_id):
        from src.domain.body_template import BodyTemplate
        return BodyTemplate(
            body_video=base / avatar_id / gesture_id / "body.mp4",
            face_mask=base / avatar_id / gesture_id / "face_mask.mp4",
            neck_mask=base / avatar_id / gesture_id / "neck_mask.mp4",
            face_transforms=base / avatar_id / gesture_id / "face_transforms.npz",
            metadata=base / avatar_id / gesture_id / "metadata.json",
        )

    timeline = Timeline.from_dict({
        "segments": [{"gesture_id": "idle", "duration_seconds": 3.0}]
    })
    with pytest.raises(ValueError) as exc:
        align_timeline(
            timeline, "alice", output_dir=tmp_path / "aligned",
            body_template_loader=loader,
        )
    assert "out-of-bounds" in str(exc.value)
    assert "width=64" in str(exc.value)


def test_align_wrong_bbox_shape_raises(tmp_path):
    base = tmp_path / "body_templates"
    base.mkdir(parents=True, exist_ok=True)
    template_dir = _synth_template(
        base, "alice", "idle", width=64, height=64,
        fps=25.0, total_frames=75,
    )
    # bbox as (N, 5) instead of (N, 4) — downstream consumers expect 4 cols.
    corrupt_bbox = np.full((75, 5), 8.0, dtype=np.float32)
    _corrupt_npz(
        template_dir,
        bbox=corrupt_bbox,
        matrices=np.tile(np.eye(4, dtype=np.float32), (75, 1, 1)),
    )

    def loader(avatar_id, gesture_id):
        from src.domain.body_template import BodyTemplate
        return BodyTemplate(
            body_video=base / avatar_id / gesture_id / "body.mp4",
            face_mask=base / avatar_id / gesture_id / "face_mask.mp4",
            neck_mask=base / avatar_id / gesture_id / "neck_mask.mp4",
            face_transforms=base / avatar_id / gesture_id / "face_transforms.npz",
            metadata=base / avatar_id / gesture_id / "metadata.json",
        )

    timeline = Timeline.from_dict({
        "segments": [{"gesture_id": "idle", "duration_seconds": 3.0}]
    })
    with pytest.raises(ValueError) as exc:
        align_timeline(
            timeline, "alice", output_dir=tmp_path / "aligned",
            body_template_loader=loader,
        )
    assert "shape" in str(exc.value)
    assert "(N, 4)" in str(exc.value)


def test_align_cross_segment_dtype_drift_normalised(tmp_path):
    base = tmp_path / "body_templates"
    base.mkdir(parents=True, exist_ok=True)
    template_a = _synth_template(
        base, "alice", "idle", width=64, height=64,
        fps=25.0, total_frames=75,
    )
    template_b = _synth_template(
        base, "alice", "explain_both", width=64, height=64,
        fps=25.0, total_frames=50,
    )
    template_c = _synth_template(
        base, "alice", "idle_back", width=64, height=64,
        fps=25.0, total_frames=75,
    )
    # Three different dtypes: float32 → float64 → float16. Align must
    # normalise the aligned output to float32 so cross-segment drift
    # does not silently upcast / downcast.
    _corrupt_npz(
        template_a,
        bbox=np.tile(
            np.array([8.0, 8.0, 56.0, 56.0], dtype=np.float32), (75, 1)
        ),
        matrices=np.tile(np.eye(4, dtype=np.float32), (75, 1, 1)),
    )
    _corrupt_npz(
        template_b,
        bbox=np.tile(
            np.array([8.0, 8.0, 56.0, 56.0], dtype=np.float64), (50, 1)
        ),
        matrices=np.tile(np.eye(4, dtype=np.float64), (50, 1, 1)),
    )
    _corrupt_npz(
        template_c,
        bbox=np.tile(
            np.array([8.0, 8.0, 56.0, 56.0], dtype=np.float16), (75, 1)
        ).astype(np.float16),
        matrices=np.tile(np.eye(4, dtype=np.float16), (75, 1, 1)),
    )

    def loader(avatar_id, gesture_id):
        from src.domain.body_template import BodyTemplate
        dir_for = {
            "idle": template_a,
            "explain_both": template_b,
            "idle_back": template_c,
        }[gesture_id]
        return BodyTemplate(
            body_video=dir_for / "body.mp4",
            face_mask=dir_for / "face_mask.mp4",
            neck_mask=dir_for / "neck_mask.mp4",
            face_transforms=dir_for / "face_transforms.npz",
            metadata=dir_for / "metadata.json",
        )

    timeline = Timeline.from_dict({
        "segments": [
            {"gesture_id": "idle", "duration_seconds": 3.0},
            {"gesture_id": "explain_both", "duration_seconds": 2.0},
            {"gesture_id": "idle_back", "duration_seconds": 3.0},
        ]
    })

    aligned = align_timeline(
        timeline, "alice", output_dir=tmp_path / "aligned",
        body_template_loader=loader,
    )

    data = np.load(str(aligned.face_transforms))
    assert data["bbox"].dtype == np.float32, (
        f"bbox must be normalised to float32 across mixed-dtype segments; "
        f"got {data['bbox'].dtype}"
    )
    assert data["matrices"].dtype == np.float32, (
        f"matrices must be normalised to float32 across mixed-dtype "
        f"segments; got {data['matrices'].dtype}"
    )
    assert data["bbox"].shape == (200, 4)
    assert data["matrices"].shape == (200, 4, 4)


def test_align_monkeypatched_non_bgr_pixel_format_raises(tmp_path, monkeypatch):
    """Defensive pixel-format check (LOW 5 follow-up).

    The default ``mp4v`` codec path used by ``_synth_template`` silently
    upconverts 1-/4-channel source frames to 3-channel BGR on read, so
    a real codec round-trip can't exercise :func:`_concatenate_videos`'s
    per-frame channel assertion. We exercise the assertion directly by
    patching :class:`cv2.VideoCapture.read` to return a 2D (H, W) frame
    on the very first read — this is the channel-shape that would trip
    the defensive check in production when a non-mp4v codec (or an
    exotic 4-channel pass-through) is loaded in.
    """
    base = tmp_path / "body_templates"
    base.mkdir(parents=True, exist_ok=True)
    _synth_template(base, "alice", "idle", width=64, height=64,
                    fps=25.0, total_frames=75)
    _synth_template(base, "alice", "explain_both", width=64, height=64,
                    fps=25.0, total_frames=50)
    _synth_template(base, "alice", "idle_back", width=64, height=64,
                    fps=25.0, total_frames=75)

    from src.domain.body_template import BodyTemplate

    def loader(avatar_id, gesture_id):
        return BodyTemplate(
            body_video=base / avatar_id / gesture_id / "body.mp4",
            face_mask=base / avatar_id / gesture_id / "face_mask.mp4",
            neck_mask=base / avatar_id / gesture_id / "neck_mask.mp4",
            face_transforms=base / avatar_id / gesture_id / "face_transforms.npz",
            metadata=base / avatar_id / gesture_id / "metadata.json",
        )

    # Patch cv2.VideoCapture.read to return a 2D (H, W) frame after the
    # first call, simulating an exotic codec pass-through.
    import src.pipeline.timeline_align as timeline_align_mod

    sentinel_iter = iter([(True, np.full((64, 64), 128, dtype=np.uint8)),
                          (False, None)] * 200)

    class StubCapture:
        """Stub of cv2.VideoCapture that fails loud on unknown methods.

        Production ``_concatenate_videos`` only calls ``isOpened``,
        ``read``, and ``release``. If a future cv2 release
        (or refactor of ``_concatenate_videos``) starts calling
        ``get`` / ``set`` / ``grab`` / ``retrieve`` / ``open`` or any
        other cv2.VideoCapture method, this stub will intentionally
        ``raise NotImplementedError`` instead of silently no-opping,
        so the test fails LOUD and the maintainer knows to either
        stub the new method or revisit the production call site.
        """

        _ALLOWED_METHODS = frozenset({"isOpened", "read", "release"})

        def __init__(self, *_args, **_kwargs):
            self._opened = True

        def isOpened(self):
            return self._opened

        def read(self):
            return next(sentinel_iter)

        def release(self):
            self._opened = False

        def __getattr__(self, name):
            if name in self._ALLOWED_METHODS:
                raise AttributeError(name)
            raise NotImplementedError(
                f"StubCapture.{name} is not implemented. "
                f"Production only calls {sorted(self._ALLOWED_METHODS)} — "
                f"if a new call site was added in _concatenate_videos, "
                f"either stub it here or update the allow-list."
            )

    monkeypatch.setattr(timeline_align_mod.cv2, "VideoCapture", StubCapture)

    timeline = Timeline.from_dict({
        "segments": [
            {"gesture_id": "idle", "duration_seconds": 3.0},
            {"gesture_id": "explain_both", "duration_seconds": 2.0},
            {"gesture_id": "idle_back", "duration_seconds": 3.0},
        ]
    })

    with pytest.raises(Exception) as exc:
        align_timeline(
            timeline, "alice", output_dir=tmp_path / "aligned",
            body_template_loader=loader,
        )
    msg = str(exc.value)
    assert "channel" in msg
    assert "BGR" in msg


# ────────────────────────────────────────────────────────────────────────────
# LOW follow-ups: dtype alignment on landmarks/confidence + ndim==4 guard
# ────────────────────────────────────────────────────────────────────────────


def test_align_cross_segment_landmarks_dtype_drift_normalised(tmp_path):
    """Test that mixed-dtype landmarks across segments are normalised
    to float32 in the aligned npz (LOW 1 follow-up #1).
    """
    base = tmp_path / "body_templates"
    base.mkdir(parents=True, exist_ok=True)
    template_a = _synth_template(
        base, "alice", "idle", width=64, height=64,
        fps=25.0, total_frames=75,
    )
    template_b = _synth_template(
        base, "alice", "explain_both", width=64, height=64,
        fps=25.0, total_frames=50,
    )
    template_c = _synth_template(
        base, "alice", "idle_back", width=64, height=64,
        fps=25.0, total_frames=75,
    )
    # Inject mixed-dtype landmarks payloads.
    _corrupt_npz(
        template_a,
        bbox=np.tile(
            np.array([8.0, 8.0, 56.0, 56.0], dtype=np.float32), (75, 1)
        ),
        matrices=np.tile(np.eye(4, dtype=np.float32), (75, 1, 1)),
    )
    _inject_array(template_a, "landmarks", _synth_landmarks(75, np.float32))
    _inject_array(template_b, "landmarks", _synth_landmarks(50, np.float64))
    _inject_array(template_c, "landmarks", _synth_landmarks(75, np.float16))

    def loader(avatar_id, gesture_id):
        from src.domain.body_template import BodyTemplate
        dir_for = {
            "idle": template_a,
            "explain_both": template_b,
            "idle_back": template_c,
        }[gesture_id]
        return BodyTemplate(
            body_video=dir_for / "body.mp4",
            face_mask=dir_for / "face_mask.mp4",
            neck_mask=dir_for / "neck_mask.mp4",
            face_transforms=dir_for / "face_transforms.npz",
            metadata=dir_for / "metadata.json",
        )

    timeline = Timeline.from_dict({
        "segments": [
            {"gesture_id": "idle", "duration_seconds": 3.0},
            {"gesture_id": "explain_both", "duration_seconds": 2.0},
            {"gesture_id": "idle_back", "duration_seconds": 3.0},
        ]
    })

    aligned = align_timeline(
        timeline, "alice", output_dir=tmp_path / "aligned",
        body_template_loader=loader,
    )

    data = np.load(str(aligned.face_transforms))
    assert data["landmarks"].dtype == np.float32, (
        f"landmarks must be normalised to float32 across mixed-dtype "
        f"segments; got {data['landmarks'].dtype}"
    )
    assert data["landmarks"].shape == (200, 478, 3)


def test_align_cross_segment_confidence_dtype_drift_normalised(tmp_path):
    """Test that mixed-dtype confidence across segments is normalised
    to float32 in the aligned npz (LOW 1 follow-up #2).
    """
    base = tmp_path / "body_templates"
    base.mkdir(parents=True, exist_ok=True)
    template_a = _synth_template(
        base, "alice", "idle", width=64, height=64,
        fps=25.0, total_frames=75,
    )
    template_b = _synth_template(
        base, "alice", "explain_both", width=64, height=64,
        fps=25.0, total_frames=50,
    )
    _inject_array(template_a, "confidence",
                  np.full((75,), 0.9, dtype=np.float32))
    _inject_array(template_b, "confidence",
                  np.full((50,), 0.85, dtype=np.float64))

    def loader(avatar_id, gesture_id):
        from src.domain.body_template import BodyTemplate
        dir_for = {"idle": template_a, "explain_both": template_b}[gesture_id]
        return BodyTemplate(
            body_video=dir_for / "body.mp4",
            face_mask=dir_for / "face_mask.mp4",
            neck_mask=dir_for / "neck_mask.mp4",
            face_transforms=dir_for / "face_transforms.npz",
            metadata=dir_for / "metadata.json",
        )

    timeline = Timeline.from_dict({
        "segments": [
            {"gesture_id": "idle", "duration_seconds": 3.0},
            {"gesture_id": "explain_both", "duration_seconds": 2.0},
        ]
    })

    aligned = align_timeline(
        timeline, "alice", output_dir=tmp_path / "aligned",
        body_template_loader=loader,
    )

    data = np.load(str(aligned.face_transforms))
    assert data["confidence"].dtype == np.float32, (
        f"confidence must be normalised to float32 across mixed-dtype "
        f"segments; got {data['confidence'].dtype}"
    )


def test_align_ndim4_frame_raises(tmp_path, monkeypatch):
    """Test that cv2.VideoCapture returning a 4D frame trips the
    explicit ndim allow-list guard (LOW 2 follow-up).
    """
    base = tmp_path / "body_templates"
    base.mkdir(parents=True, exist_ok=True)
    _synth_template(base, "alice", "idle", width=64, height=64,
                    fps=25.0, total_frames=75)
    _synth_template(base, "alice", "explain_both", width=64, height=64,
                    fps=25.0, total_frames=50)
    _synth_template(base, "alice", "idle_back", width=64, height=64,
                    fps=25.0, total_frames=75)

    from src.domain.body_template import BodyTemplate

    def loader(avatar_id, gesture_id):
        return BodyTemplate(
            body_video=base / avatar_id / gesture_id / "body.mp4",
            face_mask=base / avatar_id / gesture_id / "face_mask.mp4",
            neck_mask=base / avatar_id / gesture_id / "neck_mask.mp4",
            face_transforms=base / avatar_id / gesture_id / "face_transforms.npz",
            metadata=base / avatar_id / gesture_id / "metadata.json",
        )

    # First call returns a 4D (1, H, W, 3) frame, second call ends.
    sentinel_iter = iter([
        (True, np.full((1, 64, 64, 3), 128, dtype=np.uint8)),
        (False, None),
    ] * 200)

    import src.pipeline.timeline_align as timeline_align_mod

    class StubCapture:
        _ALLOWED_METHODS = frozenset({"isOpened", "read", "release"})

        def __init__(self, *_args, **_kwargs):
            self._opened = True

        def isOpened(self):
            return self._opened

        def read(self):
            return next(sentinel_iter)

        def release(self):
            self._opened = False

        def __getattr__(self, name):
            if name in self._ALLOWED_METHODS:
                raise AttributeError(name)
            raise NotImplementedError(
                f"StubCapture.{name} not implemented"
            )

    monkeypatch.setattr(timeline_align_mod.cv2, "VideoCapture", StubCapture)

    timeline = Timeline.from_dict({
        "segments": [
            {"gesture_id": "idle", "duration_seconds": 3.0},
            {"gesture_id": "explain_both", "duration_seconds": 2.0},
            {"gesture_id": "idle_back", "duration_seconds": 3.0},
        ]
    })

    with pytest.raises(Exception) as exc:
        align_timeline(
            timeline, "alice", output_dir=tmp_path / "aligned",
            body_template_loader=loader,
        )
    msg = str(exc.value)
    assert "ndim" in msg
    assert "unexpected tensor shape" in msg


def test_align_stubcapture_unknown_method_call_raises_not_implemented(
    tmp_path, monkeypatch
):
    """Test that StubCapture raises NotImplementedError if a future
    cv2 release / production refactor starts calling a method the
    stub does not allow (LOW 3 follow-up).
    """
    base = tmp_path / "body_templates"
    base.mkdir(parents=True, exist_ok=True)
    _synth_template(base, "alice", "idle", width=64, height=64,
                    fps=25.0, total_frames=75)

    sentinel_iter = iter([(True, np.full((64, 64), 128, dtype=np.uint8)),
                          (False, None)] * 200)

    import src.pipeline.timeline_align as timeline_align_mod

    class StubCapture:
        _ALLOWED_METHODS = frozenset({"isOpened", "read", "release"})

        def __init__(self):
            pass

        def isOpened(self):
            return True

        def read(self):
            return next(sentinel_iter)

        def release(self):
            pass

        def __getattr__(self, name):
            if name in self._ALLOWED_METHODS:
                raise AttributeError(name)
            raise NotImplementedError(
                f"StubCapture.{name} not implemented"
            )

    monkeypatch.setattr(timeline_align_mod.cv2, "VideoCapture", StubCapture)

    # Calling an unknown method directly on the stub raises NotImplementedError.
    cap = StubCapture()
    with pytest.raises(NotImplementedError) as exc:
        cap.get(cv2.CAP_PROP_FPS)
    assert "not implemented" in str(exc.value).lower()


# ────────────────────────────────────────────────────────────────────────────
# Local helpers (LOW-follow-up tests)
# ────────────────────────────────────────────────────────────────────────────


def _synth_landmarks(n_frames: int, dtype) -> np.ndarray:
    """Construct a deterministic (N, 478, 3) landmarks payload of given dtype,
    so dtype-drift regression tests have something to compare against.
    """
    base = np.zeros((478, 3), dtype=np.float32)
    base[:, 0] = np.linspace(0.1, 0.9, 478, dtype=np.float32)
    base[:, 1] = np.linspace(0.2, 0.6, 478, dtype=np.float32)
    base[:, 2] = np.full(478, 0.5, dtype=np.float32)
    arr = np.tile(base, (n_frames, 1, 1))
    if arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)
    return arr


def _inject_array(template_dir: Path, key: str, payload: np.ndarray) -> None:
    """Add ``payload`` under ``key`` to the existing face_transforms.npz
    in ``template_dir`` without disturbing the existing bbox / matrices.
    """
    existing = dict(np.load(str(template_dir / "face_transforms.npz")))
    existing[key] = payload
    np.savez_compressed(str(template_dir / "face_transforms.npz"), **existing)
