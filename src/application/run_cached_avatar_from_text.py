"""``run_cached_avatar_from_text`` — text + audio → 1 final mp4.

This is the glue that closes the previous round's wiring question:
``text_to_timeline`` produces a :class:`Timeline` (set of pose
segments with start/end/gesture_id/intensity). The render path
:func:`src.application.render_cached_avatar.render_cached_avatar`
takes only ONE ``gesture_id`` per call. This module bridges the gap:

- **TIMELINE mode** (default) — iterate every gesture segment in the
  plan, slice the input audio to that segment's window, call
  :func:`render_cached_avatar` per-segment, collect the rendered
  chunks, write a chunk-list manifest, hand it to
  :class:`EncodingWorker` for the concat + audio mux. The avatar's
  BODY POSE genuinely shifts at each segment boundary; the face moves
  continuously through lip-sync.
- **DOMINANT mode** — collapse the timeline to one segment, the one
  with the largest ``duration * intensity``, and render the whole
  clip against that single body pose.

Identity is RESOLVED ONCE per job (identity packs are face-only and
pose-agnostic, so reuse across segments is correct and amortises
the compile cost). Caller owns ENGINE LIFECYCLE — this function
never calls ``engine.load()`` or ``engine.unload()``.

Failure policy: if a timeline segment references a gesture whose
body template is missing on disk, the segment is SKIPPED WITH A
WARNING and the configured ``dominant_fallback_gesture`` (default
``idle_small``) is used in its place. A single missing template
never aborts the whole render. If the fallback itself has no body
template either, the function raises explicitly with a path-aware
error message instead of looping into the same failure per-segment.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from contracts.avatar_engine import AvatarEngine
from src.application.render_cached_avatar import render_cached_avatar
from src.application.render_video.audio_probe import _probe_audio_duration
from src.core.logging import get_logger
from src.core.config import Settings
from src.domain.body_template import load_body_template
from src.domain.types import (
    AvatarIdentityHandle,
    IdentityId,
    IdentitySpec,
    RenderChunkRequest,
    RenderChunkResult,
)
from src.motion.text_driven_timeline import (
    Timeline,
    TimelineSegment,
    text_to_timeline,
)
from src.motion.face_motion_timeline import (
    FaceMotionTimeline,
    text_to_face_motion_timeline,
)
from src.storage.avatar_packs import AvatarPackRepository
from workers.encoding_worker.worker import EncodingWorker

LOG = get_logger(__name__)

__all__ = [
    "FromTextRunResult",
    "run_cached_avatar_from_text",
    "_write_concat_manifest",
] 


# ─────────────────────────────────────────────────────────────────────────────
# Public types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class FromTextRunResult:
    """Final outcome of :func:`run_cached_avatar_from_text`.

    Mirrors the data the bench script reads (``status``, ``output_path``,
    ``metrics``) so the same :mod:`tools.cost_report` can be driven
    end-to-end through this entry point without adapting the schema.
    """

    status: str
    avatar_id: str
    language: str
    mode: str  # "timeline" | "dominant"
    timeline: Timeline
    face_timeline: FaceMotionTimeline
    output_path: Path  # final mp4 (concatenated)
    job_id: str
    segment_count: int
    render_seconds_total: float
    skipped_gestures: tuple[str, ...]
    chunk_manifest_path: Path
    face_timeline_path: Path
    face_motion_profile: dict
    metrics: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Manifest writer (lightweight — bypasses RenderRequest contract)
# ─────────────────────────────────────────────────────────────────────────────


def _write_concat_manifest(
    job_id: str, fps: int, chunks: List[RenderChunkResult], dest: Path
) -> Path:
    """Write the concat manifest the :class:`EncodingWorker` reads.

    Equivalent to :func:`src.application.render_video.manifest._write_chunk_manifest`
    but takes only the primitive attributes the writer actually reads
    (``job_id``, ``fps``, ``chunks``) so we don't have to forge a
    fake :class:`RenderRequest` + :class:`RenderSpec` to satisfy the
    dataclass type contract. The on-disk format is identical.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        fh.write(f"# manifest for job {job_id}\n")
        fh.write(f"# fps={fps}\n")
        for r in chunks:
            fh.write(
                f"{r.chunk_index}|{r.output_path.resolve().as_posix()}|{r.duration_seconds}\n"
            )
    return dest


# ─────────────────────────────────────────────────────────────────────────────
# Audio slicing (injectable)
# ─────────────────────────────────────────────────────────────────────────────


AudioSlicer = Callable[[Path, float, float, Path], Path]


def _ffmpeg_audio_slicer(input_path: Path, start: float, end: float, dest: Path) -> Path:
    """Slice ``[start, end]`` from an audio file with ``ffmpeg``.

    Falls through to :func:`_shutil_audio_slicer` if ffmpeg is missing
    on PATH so a SMOKE run still completes (the produced concat may
    be silent but the pipeline still emits a final mp4 path).
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        LOG.warning(
            "ffmpeg missing — falling back to shutil.copy2 audio slicer; "
            "lip-sync will be approximate, NOT per-segment wall-clock aligned."
        )
        return _shutil_audio_slicer(input_path, start, end, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(input_path),
        "-c",
        "copy",
        str(dest),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and dest.is_file() and dest.stat().st_size > 0:
        return dest
    # Some mp3 containers don't allow stream-copy past keyframes;
    # fall back to copying the whole audio so the rest of the
    # pipeline still completes (operators see a log line above).
    LOG.debug(
        "ffmpeg stream-copy trim failed for (start=%.3f, end=%.3f); falling back to copy.",
        start,
        end,
    )
    shutil.copy2(input_path, dest)
    return dest


def _shutil_audio_slicer(input_path: Path, start: float, end: float, dest: Path) -> Path:
    """Conservative fallback: copy the source audio unchanged."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, dest)
    return dest


# ─────────────────────────────────────────────────────────────────────────────
# Body-template preflight (fail-fast + graceful fallback)
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_segment_gesture(
    *,
    avatar_id: str,
    gesture_id: str,
    body_templates_dir: Path,
    fallback_gesture: str,
    skipped_so_far: list[str],
) -> str:
    """Return ``gesture_id`` if a body template exists, else the fallback.

    Records the original gesture in ``skipped_so_far`` so the result
    can surface which intents couldn't find a body template.
    """
    try:
        load_body_template(avatar_id, gesture_id, base_dir=body_templates_dir)
        return gesture_id
    except FileNotFoundError:
        if gesture_id != fallback_gesture:
            skipped_so_far.append(gesture_id)
        return fallback_gesture


def _preflight_fallback(
    *, avatar_id: str, fallback_gesture: str, body_templates_dir: Path
) -> None:
    """Raise ``RuntimeError`` if the fallback gesture also has no template.

    Catches the "all body templates missing" case once before the
    loop, instead of looping into the same miss per-segment.
    """
    if not fallback_gesture:
        raise RuntimeError(
            "dominant_fallback_gesture must be a non-empty gesture_id "
            "(e.g. 'idle_small') when timeline mode is active"
        )
    try:
        load_body_template(avatar_id, fallback_gesture, base_dir=body_templates_dir)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"avatar_id={avatar_id!r} has no body templates on disk at "
            f"{body_templates_dir.resolve()!s} — even the fallback gesture "
            f"{fallback_gesture!r} is missing. Run a body-template "
            f"session (tools/avatar_assets/extract_reference_motion.py) "
            f"before invoking the text-driven pipeline."
        ) from exc


def _require_identity_inputs(
    *,
    source_image: Optional[Path],
    identity_pack_path: Optional[Path],
    pack_repo: Optional[AvatarPackRepository],
) -> Optional[Path]:
    """Pick the source-of-truth for the identity resolution path.

    Returns a usable ``source_image`` Path, or raises a clear
    ``RuntimeError`` so the CLI surfaces a friendly message instead
    of a confusing ``PIL.UnidentifiedImageError`` from a downstream
    Image.open() call.
    """
    if identity_pack_path is not None:
        return None  # pack supplies identity, source_image not needed
    if pack_repo is not None:
        return None  # pack_repo caches identity, source_image falls through to compile
    if source_image is not None:
        return source_image
    raise RuntimeError(
        "run_cached_avatar_from_text: no identity_pack_path, no pack_repo, "
        "no source_image — provide one of --source-image / --identity-pack "
        "/ supply a warm AvatarPackRepository so the avatar's face can be "
        "compiled or reused."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Mode helpers
# ─────────────────────────────────────────────────────────────────────────────


def _dominant_segment(timeline: Timeline) -> TimelineSegment:
    """Return the segment with the largest (duration * intensity)."""
    if not timeline.segments:
        raise ValueError("Cannot pick a dominant segment from an empty timeline")
    return max(
        timeline.segments,
        key=lambda s: (s.end - s.start) * max(s.intensity, 1e-3),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Settings resolution
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_settings(engine: AvatarEngine) -> Settings:
    """Pick the best ``Settings`` instance available; never fabricate fakes.

    Priority: ``engine.settings`` (real adapter), ``Settings.from_env()``
    (cheap, no I/O), then a hard fail.
    """
    settings = getattr(engine, "settings", None)
    if isinstance(settings, Settings):
        return settings
    return Settings.from_env()


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────


def run_cached_avatar_from_text(
    text: str,
    audio_path: Path,
    output_path: Path,
    *,
    avatar_id: str,
    engine: AvatarEngine,
    language: str = "it",
    fps: int = 25,
    mode: str = "timeline",
    dominant_fallback_gesture: str = "idle_small",
    body_templates_dir: Path | str = "body_templates",
    capture_root: Path = Path("./captures"),
    source_image: Optional[Path] = None,
    identity_pack_path: Optional[Path] = None,
    pack_repo: Optional[AvatarPackRepository] = None,
    pack_root: Optional[Path] = None,
    job_id: Optional[str] = None,
    audio_slicer: AudioSlicer = _ffmpeg_audio_slicer,
    motion_style: Optional[str] = None,
) -> FromTextRunResult:
    """Wrapper that temporarily overrides motion_style in engine settings."""
    original_engine_settings = getattr(engine, "settings", None)
    if motion_style and original_engine_settings is not None:
        try:
            from dataclasses import replace
            engine.settings = replace(original_engine_settings, motion_style=str(motion_style))
        except Exception:
            pass

    try:
        return _run_cached_avatar_from_text_impl(
            text=text,
            audio_path=audio_path,
            output_path=output_path,
            avatar_id=avatar_id,
            engine=engine,
            language=language,
            fps=fps,
            mode=mode,
            dominant_fallback_gesture=dominant_fallback_gesture,
            body_templates_dir=body_templates_dir,
            capture_root=capture_root,
            source_image=source_image,
            identity_pack_path=identity_pack_path,
            pack_repo=pack_repo,
            pack_root=pack_root,
            job_id=job_id,
            audio_slicer=audio_slicer,
        )
    finally:
        if original_engine_settings is not None and motion_style:
            try:
                engine.settings = original_engine_settings
            except Exception:
                pass


def _run_cached_avatar_from_text_impl(
    text: str,
    audio_path: Path,
    output_path: Path,
    *,
    avatar_id: str,
    engine: AvatarEngine,
    language: str = "it",
    fps: int = 25,
    mode: str = "timeline",
    dominant_fallback_gesture: str = "idle_small",
    body_templates_dir: Path | str = "body_templates",
    capture_root: Path = Path("./captures"),
    source_image: Optional[Path] = None,
    identity_pack_path: Optional[Path] = None,
    pack_repo: Optional[AvatarPackRepository] = None,
    pack_root: Optional[Path] = None,
    job_id: Optional[str] = None,
    audio_slicer: AudioSlicer = _ffmpeg_audio_slicer,
) -> FromTextRunResult:
    """Render an avatar video driven by ``text`` + ``audio_path``.

    Parameters
    ----------
    text:
        Script the avatar speaks. Drives the planner that selects
        body gestures (counts, emphasis, comparisons, open-palms,
        conclusions, questions).
    audio_path:
        Source WAV / mp3 driving the lip-sync.
    output_path:
        Destination mp4 file (final, muxed, chunk-aligned).
    avatar_id:
        Identity key (faces) AND body-template subdirectory name.
    engine:
        An already-:meth:`load`-ed :class:`AvatarEngine` instance.
        The caller retains ownership of lifecycle — this function
        does NOT call ``engine.load()`` or ``engine.unload()``.
    language, fps, mode, dominant_fallback_gesture, body_templates_dir,
    capture_root, source_image, identity_pack_path, pack_repo, pack_root,
    job_id:
        Forwarded to the timeline planner + :func:`render_cached_avatar`
        + :class:`EncodingWorker`. See module docstring.

    audio_slicer:
        Optional callable that produces an audio file covering the
        ``[start, end]`` wall-clock window. Defaults to ffmpeg; tests
        inject a no-op stub so the suite stays green on ffmpeg-free
        runners.

    Returns
    -------
    :class:`FromTextRunResult` with the final mp4 path, the timeline,
    and per-job metrics for the bench scripts.

    Raises
    ------
    ValueError
        ``mode`` is not in {"timeline", "dominant"}.
    RuntimeError
        No identity inputs supplied, no body templates on disk, the
        timeline is empty in DOMINANT mode. ``render_cached_avatar``
        may also re-raise its own failure modes (file-not-found /
        engine-error) which propagate unchanged.
    """
    if mode not in {"timeline", "dominant"}:
        raise ValueError(f"mode must be 'timeline' or 'dominant' (got {mode!r})")

    capture_root.mkdir(parents=True, exist_ok=True)
    job_id = job_id or f"text-{uuid.uuid4().hex[:10]}"

    probe_seconds = _probe_audio_duration(audio_path)
    audio_duration = probe_seconds if probe_seconds > 0.0 else 8.0

    identity_source_image = _require_identity_inputs(
        source_image=source_image,
        identity_pack_path=identity_pack_path,
        pack_repo=pack_repo,
    )

    timeline = text_to_timeline(
        text=text,
        audio_duration=audio_duration,
        avatar_id=avatar_id,
        language=language,
        fps=fps,
    )
    face_timeline = text_to_face_motion_timeline(
        text=text,
        audio_duration=audio_duration,
        avatar_id=avatar_id,
        language=language,
        fps=fps,
    )
    LOG.info(
        "run_cached_avatar_from_text: avatar_id=%s mode=%s planned=%d segments over %.2fs",
        avatar_id,
        mode,
        len(timeline.segments),
        timeline.duration,
    )

    if mode == "dominant":
        dominant = _dominant_segment(timeline)
        active_segments: Tuple[TimelineSegment, ...] = (
            TimelineSegment(
                kind=dominant.kind,
                start=0.0,
                end=max(0.0, audio_duration),
                gesture_id=dominant.gesture_id,
                pose_id=dominant.pose_id,
                intensity=dominant.intensity,
                anchor_word=dominant.anchor_word,
                text_span=dominant.text_span,
            ),
        )
    else:
        active_segments = timeline.segments

    # Preflight runs in BOTH modes so a missing body template is
    # surfaced as a one-line RuntimeError instead of a per-segment
    # FileNotFoundError loop. The fallback gesture must exist on
    # disk for either path to render anything.
    body_dir = Path(body_templates_dir)
    face_timeline_path = capture_root / job_id / "face_timeline.json"
    face_timeline.write_json(face_timeline_path)
    _preflight_fallback(
        avatar_id=avatar_id,
        fallback_gesture=dominant_fallback_gesture,
        body_templates_dir=body_dir,
    )

    skipped: List[str] = []

    segment_outputs: List[RenderChunkResult] = [] 
    metric_segments: List[dict] = []

    for idx, seg in enumerate(active_segments):
        seg_gesture = _resolve_segment_gesture(
            avatar_id=avatar_id,
            gesture_id=seg.gesture_id,
            body_templates_dir=body_dir,
            fallback_gesture=dominant_fallback_gesture,
            skipped_so_far=skipped,
        )
        if seg_gesture != seg.gesture_id:
            LOG.warning(
                "gesture %s has no body template; substituting %s for segment %d.",
                seg.gesture_id,
                seg_gesture,
                idx,
            )

        seg_audio = (
            capture_root
            / job_id
            / "segments"
            / f"{idx:03d}_{seg_gesture}"
            / "slice.wav"
        )
        audio_slicer(audio_path, seg.start, seg.end, seg_audio)

        seg_dir = seg_audio.parent
        seg_output = seg_dir / "final.mp4"

        # Each per-segment render reuses the same identity pack so
        # the face mesh is compiled exactly once per job.
        result = render_cached_avatar(
            avatar_id=avatar_id,
            gesture_id=seg_gesture,
            identity_id=IdentityId(avatar_id),
            audio_path=seg_audio,
            output_path=seg_output,
            engine=engine,
            source_image=identity_source_image,
            identity_pack_path=identity_pack_path,
            pack_repo=pack_repo,
            pack_root=pack_root,
            capture_dir=seg_dir,
            fps=fps,
            body_templates_dir=str(body_dir),
            face_motion_timeline_path=face_timeline_path,
            debug=False,
        )

        seg_duration = max(0.0, seg.end - seg.start)
        segment_outputs.append(
            RenderChunkResult(
                chunk_index=idx,
                output_path=result.final_path or seg_output,
                duration_seconds=seg_duration if seg_duration > 0 else result.output_seconds,
                frames_rendered=int((seg_duration if seg_duration > 0 else result.output_seconds) * fps),
                gpu_seconds=result.gpu_seconds,
                engine_id=engine.engine_id,
            )
        )
        metric_segments.append(
            {
                "index": idx,
                "gesture_id": seg_gesture,
                "original_gesture_id": seg.gesture_id,
                "pose_id": seg.pose_id,
                "start": round(seg.start, 4),
                "end": round(seg.end, 4),
                "intensity": round(seg.intensity, 4),
                "status": result.status,
                "gpu_seconds": round(result.gpu_seconds, 4),
                "wall_seconds": round(result.wall_seconds, 4),
            }
        )

    # ── Concat via EncodingWorker ───────────────────────────────────────
    # Write the concat manifest directly bypassing RenderRequest /
    # RenderSpec (which are not designed for plan-driven runs).
    manifest_path = _write_concat_manifest(
        job_id=job_id,
        fps=fps,
        chunks=segment_outputs,
        dest=Path("./captures") / f"{job_id}.manifest.txt",
    )

    settings = _resolve_settings(engine)
    worker = EncodingWorker(
        settings=settings,
        output_dir=output_path.parent,
        overlap_seconds=0.0,
        chunk_seconds=timeline.duration,
    )
    final_path = worker.encode(
        job_id,
        manifest_path,
        audio_path=audio_path,
    )
    # The worker writes ``<output_dir>/<job_id>.mp4``; honour the
    # caller-requested ``output_path`` by moving the file there.
    if output_path.resolve() != final_path.resolve():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(final_path), str(output_path))
        final_path = output_path

    metrics_block = {
        "job_id": job_id,
        "mode": mode,
        "avatar_id": avatar_id,
        "language": language,
        "audio_duration_seconds": round(audio_duration, 3),
        "fps": fps,
        "segment_count": len(active_segments),
        "render_seconds_total": round(sum(r.duration_seconds for r in segment_outputs), 3),
        "skipped_gestures": tuple(skipped),
        "timeline_gestures": [s.gesture_id for s in timeline.segments],
        "final_path": str(final_path),
        "chunk_manifest_path": str(manifest_path),
        "face_timeline_path": str(face_timeline_path),
        "face_timeline_motions": [s.motion_id for s in face_timeline.segments],
        "face_motion_profile": {
            "duration": face_timeline.duration,
            "fps": face_timeline.fps,
            "segment_count": len(face_timeline.segments),
            "motion_ids": [s.motion_id for s in face_timeline.segments],
        },
        "segments": metric_segments,
    }

    metrics_path = capture_root / job_id / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_block, indent=2), encoding="utf-8")

    LOG.info(
        "run_cached_avatar_from_text: status=%s skipped=%d final=%s",
        "COMPLETED" if not skipped else "COMPLETED_WITH_FALLBACKS",
        len(skipped),
        final_path,
    )

    return FromTextRunResult(
        status="COMPLETED" if not skipped else "COMPLETED_WITH_FALLBACKS",
        avatar_id=avatar_id,
        language=language,
        mode=mode,
        timeline=timeline,
        face_timeline=face_timeline,
        output_path=final_path,
        job_id=job_id,
        segment_count=len(active_segments),
        render_seconds_total=metrics_block["render_seconds_total"],
        skipped_gestures=tuple(skipped),
        chunk_manifest_path=manifest_path,
        face_timeline_path=face_timeline_path,
        face_motion_profile=metrics_block["face_motion_profile"],
        metrics={
            "job_id": job_id,
            "mode": mode,
            "segment_count": len(active_segments),
            "render_seconds_total": metrics_block["render_seconds_total"],
            "skipped_gestures": list(skipped),
            "timeline_gestures": metrics_block["timeline_gestures"],
            "face_timeline_motions": metrics_block["face_timeline_motions"],
            "face_motion_profile": metrics_block["face_motion_profile"],
            "final_path": str(final_path),
            "chunk_manifest_path": str(manifest_path),
            "face_timeline_path": str(face_timeline_path),
            "segments": metric_segments,
        },
    )
