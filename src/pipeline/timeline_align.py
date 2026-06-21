"""Frame-aligned body timeline materialiser for the multi-template pipeline.

Per ``docs/REPOSITORY_SLIMMING_PLAN.md`` §6 + §10 (Change 4 of the
slim plan), the runtime MUST keep video frames / face transforms /
face masks / neck masks / timestamps / audio perfectly aligned across
segment boundaries BEFORE the orchestrator hands them to MuseTalk
and the compositor. This module is the concrete materialiser for
that property.

What it does
------------
:func:`align_timeline` parses a :class:`src.domain.timeline.Timeline`,
loads each segment's :class:`src.domain.body_template.BodyTemplate`,
validates the cross-segment invariants (frame count, resolution,
fps), and writes FOUR concatenated files on disk:

* ``body.mp4``             — one continuous body video at Timeline fps.
* ``face_mask.mp4``        — one continuous face-mask video.
* ``neck_mask.mp4``        — one continuous neck-mask video.
* ``face_transforms.npz``  — concatenated bbox + matrices +
  optional landmarks + confidence + remapped timestamp_ms arrays.
* ``metadata.json``        — segment-level duration + frame breakdown
  so the orchestrator / QC layer can audit provenance.

The output :class:`AlignedBodyTimeline` exposes the same 4-file
shape as a single :class:`src.domain.body_template.BodyTemplate` so
downstream code (the canonical compositor + the split-importer
tests in the package surface) does not need to know it came from
multiple source clips.

Strict invariants
-----------------
The align utility is deliberately STINGY about what it accepts. The
slim plan's acceptance bullet "video, masks, and transform arrays
remain frame-aligned" means a frame count off by one is a real
bug, not a tolerance band. The following conditions trigger an
explicit :class:`ValueError`:

* Per-segment frame count != ``round(segment.duration_seconds * fps)``.
* Width or height disagree across segments.
* Template-recorded fps disagrees with ``timeline.fps``.
* ``face_transforms.npz`` is missing ``bbox`` or ``matrices``.
* ``bbox`` array contains ``NaN``/``inf`` values OR has any row with
  coordinates outside the validated image bounds.
* ``bbox`` is not exactly shape ``(N, 4)``; ``matrices`` is not
  exactly shape ``(N, 4, 4)``.
* A source ``<kind>.mp4`` frame has a non-BGR pixel format (anything
  other than 3 channels).
* A template file is missing on disk.

Output standardisation (silent corruption guards):

* ``bbox``, ``matrices``, ``landmarks`` (if present) and
  ``confidence`` (if present) are all coerced to ``float32`` so
  cross-segment dtype drift cannot double the aligned-npz size or
  silently change precision downstream. The sub-mm precision
  degradation on the optional ``landmarks`` (478×3 3D positions)
  is negligible at the canonical Timeline resolution.
* ``timestamp_ms`` is rebuilt as a strictly-monotonic viewport-time
  sequence (``1000 / fps`` ms dt, cumulative segment offsets) — NOT a
  per-template timestamp concat. The precompute tool's per-template
  semantic timestamps are discarded on purpose (see _concatenate_npz
  docstring).

The downstream orchestrator does NOT need to defensively re-check
these; it can trust the AlignedBodyTimeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np

from src.core.logging import get_logger
from src.domain.body_template import BodyTemplate
from src.domain.timeline import Timeline
from src.quality.exceptions import CompositeError

LOG = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class AlignedBodyTimeline:
    """Concatenated body timeline; same shape as :class:`BodyTemplate`.

    The ``metadata`` field is computed at call time as a derived
    path (``body_video.with_suffix(".metadata.json")``) so iterating
    the aligned dir exposes the segment-level breakdown alongside
    the canonical files.
    """

    body_video: Path
    face_mask: Path
    neck_mask: Path
    face_transforms: Path
    metadata: Path

    total_frames: int
    fps: int
    duration_seconds: float
    width: int
    height: int

    def as_body_template(self) -> BodyTemplate:
        """Re-expose as a single :class:`BodyTemplate` for downstream code."""
        return BodyTemplate(
            body_video=self.body_video,
            face_mask=self.face_mask,
            neck_mask=self.neck_mask,
            face_transforms=self.face_transforms,
            metadata=self.metadata,
        )


def align_timeline(
    timeline: Timeline,
    avatar_id: str,
    *,
    output_dir: Path | str,
    fps: int | None = None,
    body_template_loader=None,
) -> AlignedBodyTimeline:
    """Concatenate ``timeline`` into one frame-aligned body timeline.

    Args:
        timeline: ordered segments + canonical fps (from
            ``src.domain.timeline.Timeline``).
        avatar_id: avatar whose ``body_templates/<avatar_id>/<gesture_id>/``
            tree holds the source clips. Convention enforces the
            same layout that ``precompute_video_template.py`` writes.
        output_dir: directory in which the four aligned files plus
            ``metadata.json`` are materialised. Will be created.
        fps: optional override for ``timeline.fps``; defaults to
            ``timeline.fps``. Kept for ops-driven deployment pinning.
        body_template_loader: injectable for tests. Defaults to
            :func:`src.domain.body_template.load_body_template`.

    Returns:
        :class:`AlignedBodyTimeline` describing the four written files
        + computed totals.

    Raises:
        ValueError: frame count or fps or dimension invariant violation.
        FileNotFoundError: a segment's source files are missing on disk.
        CompositeError: cv2 fails to open a VideoCapture / VideoWriter
            on the aligned path.
    """
    if body_template_loader is None:
        from src.domain.body_template import load_body_template

        body_template_loader = load_body_template

    effective_fps = int(fps or timeline.fps)
    if effective_fps <= 0:
        raise ValueError(
            f"align_timeline: fps must be positive; got {effective_fps}"
        )
    fps = effective_fps

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    templates: List[BodyTemplate] = []
    expected_frame_counts: List[int] = []
    for idx, seg in enumerate(timeline.segments):
        template = body_template_loader(avatar_id, seg.gesture_id)
        expected = timeline.frame_count_for_segment(idx)
        actual = template.frame_count()
        if actual != expected:
            raise ValueError(
                f"align_timeline: frame-count mismatch on segment "
                f"{idx} (avatar_id={avatar_id!r}, gesture_id="
                f"{seg.gesture_id!r}): template has {actual} frames "
                f"but timeline prescribes {expected} "
                f"(=round({seg.duration_seconds} * {fps})). "
                f"Re-precompute the body template at the correct "
                f"duration/fps and retry."
            )
        templates.append(template)
        expected_frame_counts.append(expected)

    width = templates[0].width()
    height = templates[0].height()
    template_fps = templates[0].fps()
    for idx, t in enumerate(templates):
        if t.width() != width or t.height() != height:
            raise ValueError(
                f"align_timeline: dimension mismatch on segment {idx} "
                f"(gesture_id={t.gesture_id()!r}): "
                f"({t.width()}x{t.height()}) ≠ first segment "
                f"({width}x{height}). All segments must share one "
                f"resolution."
            )
        if abs(t.fps() - template_fps) > 1e-6:
            raise ValueError(
                f"align_timeline: cross-template fps disagreement on "
                f"segment {idx} (gesture_id={t.gesture_id()!r}): "
                f"{t.fps()} ≠ {template_fps}"
            )
    if abs(template_fps - fps) > 1e-6:
        raise ValueError(
            f"align_timeline: timeline fps={fps} disagrees with "
            f"precomputed template fps={template_fps}"
        )

    total_frames = sum(expected_frame_counts)

    body_out = output_dir / "body.mp4"
    fmask_out = output_dir / "face_mask.mp4"
    nmask_out = output_dir / "neck_mask.mp4"
    npz_out = output_dir / "face_transforms.npz"
    metadata_out = output_dir / "metadata.json"

    _concatenate_videos(
        sources=[t.body_video for t in templates],
        destination=body_out,
        width=width, height=height, fps=fps,
        kind="body",
    )
    _concatenate_videos(
        sources=[t.face_mask for t in templates],
        destination=fmask_out,
        width=width, height=height, fps=fps,
        kind="face_mask",
    )
    _concatenate_videos(
        sources=[t.neck_mask for t in templates],
        destination=nmask_out,
        width=width, height=height, fps=fps,
        kind="neck_mask",
    )
    _concatenate_npz(
        templates=templates,
        destination=npz_out,
        expected_frame_counts=expected_frame_counts,
        fps=fps,
        width=width,
        height=height,
    )

    metadata_payload = {
        "avatar_id": avatar_id,
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "duration_seconds": float(total_frames / fps),
        "timeline_total_duration_seconds": timeline.total_duration_seconds(),
        "segments": [
            {
                "gesture_id": s.gesture_id,
                "duration_seconds": s.duration_seconds,
                "frames": expected_frame_counts[i],
            }
            for i, s in enumerate(timeline.segments)
        ],
        "status": "aligned",
    }
    with open(metadata_out, "w", encoding="utf-8") as fh:
        json.dump(metadata_payload, fh, indent=2)

    LOG.info(
        "Align timeline @%d fps: avatar=%s segments=%d -> %d frames, "
        "%.2fs, written to %s",
        fps,
        avatar_id,
        len(templates),
        total_frames,
        float(total_frames / fps),
        output_dir,
    )

    return AlignedBodyTimeline(
        body_video=body_out,
        face_mask=fmask_out,
        neck_mask=nmask_out,
        face_transforms=npz_out,
        metadata=metadata_out,
        total_frames=total_frames,
        fps=fps,
        duration_seconds=float(total_frames / fps),
        width=width,
        height=height,
    )


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────


def _concatenate_videos(
    *,
    sources: Sequence[Path],
    destination: Path,
    width: int,
    height: int,
    fps: int,
    kind: str,
) -> None:
    """Cascade N source mp4s into ``destination`` frame-by-frame.

    The destination is re-encoded at ``(width, height, fps)`` so the
    align product is one stream; pixel data is copied verbatim from
    the source (cv2.VideoCapture writes / decodes the same BGR layout
    ``precompute_video_template.py`` produces).

    Pixel-format continuity is asserted per frame: only 3-channel BGR
    frames are accepted. Gray (1-channel) or RGBA (4-channel) source
    mp4s would cause cv2 to silently down/up-mix channels depending
    on build, so a 1/4-channel frame triggers an explicit
    :class:`CompositeError` rather than corrupting the aligned stream.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(destination), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise CompositeError(
            f"align_timeline: cannot open VideoWriter for "
            f"{kind}={destination}"
        )
    try:
        for src in sources:
            cap = cv2.VideoCapture(str(src))
            if not cap.isOpened():
                raise FileNotFoundError(
                    f"align_timeline: cannot open {kind} source {src}"
                )
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    nd = frame.ndim
                    # Allow-list shape: cv2.VideoCapture.read() on
                    # colour sources returns a 3D (H, W, C) array; on
                    # grayscale sources a 2D (H, W) array. Anything
                    # else (a 4D batched array, a 1D scalar, a wraper
                    # returning a list) is a misconfigured source —
                    # reject up-front rather than silently indexing
                    # into the wrong axis.
                    if nd not in (2, 3):
                        raise CompositeError(
                            f"align_timeline: {kind} frame on {src} "
                            f"has ndim={nd}; expected 2 (grayscale) "
                            f"or 3 (BGR). cv2.VideoCapture is "
                            f"returning an unexpected tensor shape."
                        )
                    h_, w_ = frame.shape[0], frame.shape[1]
                    channels = frame.shape[2] if nd == 3 else 1
                    if w_ != width or h_ != height:
                        raise CompositeError(
                            f"align_timeline: {kind} frame shape "
                            f"mismatch on {src} (got "
                            f"{w_}x{h_}, expected {width}x{height})"
                        )
                    if channels != 3:
                        raise CompositeError(
                            f"align_timeline: {kind} frame on {src} "
                            f"has {channels} channel(s); expected 3 "
                            f"(BGR). Re-render the source mp4 at "
                            f"3-channel BGR and retry."
                        )
                    writer.write(frame)
            finally:
                cap.release()
    finally:
        writer.release()


def _concatenate_npz(
    *,
    templates: Sequence[BodyTemplate],
    destination: Path,
    expected_frame_counts: Sequence[int],
    fps: int,
    width: int,
    height: int,
) -> None:
    """Concatenate ``face_transforms.npz`` arrays across templates.

    The ``bbox`` (N_total, 4) and ``matrices`` (N_total, 4, 4) arrays
    are concatenated along axis 0; ``landmarks`` (N_total, 478, 3)
    and ``confidence`` (N_total,) if present in the source follow the
    same pattern. ``timestamp_ms`` (N_total,) is REBUILT as a
    strictly-monotonic viewport-time sequence (``1000 / fps`` ms dt
    starting at 0, with cumulative segment offsets) — NOT a verbatim
    concat of the source per-template timestamps, so it reads as if
    the entire timeline had been precomputed at the canonical fps in
    a single pass. This drops the precompute tool's per-template
    semantic timestamp jitter; the rebuild is what the real (single
    precomputed) face_transforms.npz would contain.

    Hard validation per source template (raises :class:`ValueError`):

    * ``bbox`` / ``matrices`` length aligns with metadata frames.
    * ``bbox`` is exactly shape ``(N, 4)``; ``matrices`` is exactly
      shape ``(N, 4, 4)``.
    * ``bbox`` and ``matrices`` contain no ``NaN``/``inf`` values.
    * ``bbox`` rows satisfy ``x_min >= 0``, ``y_min >= 0``,
      ``x_max <= width``, ``y_max <= height``.

    Output standardisation (silent-corruption guards):

    * ``bbox``, ``matrices``, ``landmarks`` (if present) and
      ``confidence`` (if present) are all coerced to ``float32`` so
      cross-segment dtype drift (e.g. one segment's ``landmarks``
      stored as float32 + the next as float64) cannot silently
      upcast the aligned npz and double its on-disk footprint. The
      sub-mm precision loss on ``landmarks`` (478×3 3D positions)
      is negligible at the canonical Timeline resolution.
    """
    bbox_parts: List[np.ndarray] = []
    mat_parts: List[np.ndarray] = []
    lmk_parts: List[np.ndarray] = []
    conf_parts: List[np.ndarray] = []
    ts_parts: List[np.ndarray] = []

    cumulative_frames = 0
    ms_per_frame = int(round(1000.0 / fps))
    for t, n_frames in zip(templates, expected_frame_counts):
        data = np.load(str(t.face_transforms))
        bbox = data.get("bbox", data.get("bboxes"))
        matrices = data.get("matrices")
        if bbox is None or matrices is None:
            raise ValueError(
                f"align_timeline: face_transforms.npz at "
                f"{t.face_transforms} missing required keys "
                f"'bbox' and/or 'matrices'"
            )
        bbox_arr = np.asarray(bbox)
        matrices_arr = np.asarray(matrices)
        if bbox_arr.shape[0] != n_frames:
            raise ValueError(
                f"align_timeline: bbox length {bbox_arr.shape[0]} "
                f"disagrees with metadata frames={n_frames} on "
                f"{t.face_transforms}"
            )
        if matrices_arr.shape[0] != n_frames:
            raise ValueError(
                f"align_timeline: matrices length "
                f"{matrices_arr.shape[0]} disagrees with metadata "
                f"frames={n_frames} on {t.face_transforms}"
            )
        if bbox_arr.ndim != 2 or bbox_arr.shape[1] != 4:
            raise ValueError(
                f"align_timeline: bbox shape "
                f"{tuple(bbox_arr.shape)} on {t.face_transforms} "
                f"disagrees with required shape (N, 4)"
            )
        if matrices_arr.ndim != 3 or matrices_arr.shape[1:] != (4, 4):
            raise ValueError(
                f"align_timeline: matrices shape "
                f"{tuple(matrices_arr.shape)} on {t.face_transforms} "
                f"disagrees with required shape (N, 4, 4)"
            )
        if not np.isfinite(bbox_arr).all():
            raise ValueError(
                f"align_timeline: bbox on {t.face_transforms} "
                f"contains non-finite (NaN/inf) values"
            )
        if not np.isfinite(matrices_arr).all():
            raise ValueError(
                f"align_timeline: matrices on {t.face_transforms} "
                f"contains non-finite (NaN/inf) values"
            )
        if (
            (bbox_arr[:, 0] < 0).any()
            or (bbox_arr[:, 1] < 0).any()
            or (bbox_arr[:, 2] > width).any()
            or (bbox_arr[:, 3] > height).any()
        ):
            raise ValueError(
                f"align_timeline: bbox on {t.face_transforms} has "
                f"out-of-bounds coordinates (width={width}, "
                f"height={height}). Re-precompute the body template "
                f"and retry."
            )
        # Normalise dtype so cross-segment drift cannot silently
        # upcast the aligned npz (e.g. one segment float32 + next
        # float64 → np.concatenate → float64 for the whole stack).
        bbox_parts.append(bbox_arr.astype(np.float32, copy=False))
        mat_parts.append(matrices_arr.astype(np.float32, copy=False))

        landmarks = data.get("landmarks")
        if landmarks is not None:
            lmk_arr = np.asarray(landmarks)
            if lmk_arr.shape[0] != n_frames:
                raise ValueError(
                    f"align_timeline: landmarks length "
                    f"{lmk_arr.shape[0]} disagrees with metadata "
                    f"frames={n_frames} on {t.face_transforms}"
                )
            # Normalise dtype to float32 to match bbox / matrices
            # canonical dtype and prevent cross-segment drift from
            # silently upcasting the aligned npz.
            lmk_parts.append(lmk_arr.astype(np.float32, copy=False))
        confidences = data.get("confidence")
        if confidences is not None:
            conf_arr = np.asarray(confidences)
            if conf_arr.shape[0] != n_frames:
                raise ValueError(
                    f"align_timeline: confidence length "
                    f"{conf_arr.shape[0]} disagrees with metadata "
                    f"frames={n_frames} on {t.face_transforms}"
                )
            # Normalise dtype to float32 — same rationale as
            # landmarks / bbox / matrices.
            conf_parts.append(conf_arr.astype(np.float32, copy=False))
        # Viewport-time remap (NOT per-template timestamp concat):
        # frame 0 of segment K = cumulative_frames * ms_per_frame,
        # where cumulative_frames tracks the running total before this
        # segment. Loses the precompute tool's per-template dt jitter
        # by design — the align product is what a single precomputed
        # body_template.BodyTemplate of the canonical timeline would
        # have written.
        ts_parts.append(
            np.arange(n_frames, dtype=np.int64) * ms_per_frame
            + cumulative_frames * ms_per_frame
        )
        cumulative_frames += n_frames

    save_kwargs: dict = {
        "bbox": np.concatenate(bbox_parts, axis=0),
        "matrices": np.concatenate(mat_parts, axis=0),
        "timestamp_ms": np.concatenate(ts_parts, axis=0),
    }
    if lmk_parts:
        save_kwargs["landmarks"] = np.concatenate(lmk_parts, axis=0)
    if conf_parts:
        save_kwargs["confidence"] = np.concatenate(conf_parts, axis=0)

    np.savez_compressed(str(destination), **save_kwargs)


__all__ = ["AlignedBodyTimeline", "align_timeline"]
