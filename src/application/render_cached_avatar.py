"""``render_cached_avatar`` — the economical one-call use case.

Single entry point that wires together every stage of the *cached*
pipeline described in ``docs/REPOSITORY_SLIMMING_PLAN.md`` §6:

1. ``load_body_template``      — fail-closed prerecorded body clip.
2. Resolve Identity            — load pre-built pack or compile fresh.
3. ``extract_face_roi``        — 256×256 ROI crop from the body clip.
4. ``engine.render_chunk``     — MuseTalk face-region-only inference.
5. ``OpenCVFaceCompositor``    — paste-back, feathering, color match.
6. ``mux_audio``               — ffmpeg -shortest to rejoin the WAV.
7. ``VideoQualityChecker``     — six-point QC gate.

This is the use case called by the "Block 2" benchmark in the project
plan: pure ``body_template + face ROI + MuseTalk + compositor + QC``,
no per-video body recompile or MediaPipe face detection.

The result dataclass records the canonical block-1 metrics the week-3
plan asks for: ``body_cache_hit``, ``identity_cache_hit``,
``model_warm``, ``face_region_only``, ``batch_size``,
``gpu_seconds`` (real, measured), and ``wall_seconds``.

Notes for callers
-----------------
* Mock mode (``HEYAVATAR_MOCK_ENGINE=1``) still works: the engine
  emits synthetic 256×256 mp4 for the face region so the compositor +
  QC stages exercise the same code paths as production.
* ``engine`` is injectable so tests / cross-engine benchmarks can
  swap adapters (LivePortrait vs. MuseTalk).
* An explicit ``identity_pack_path`` short-circuits the compile step
  for cache-hit measurement.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from contracts.avatar_engine import AvatarEngine, EngineState
from contracts.compositor import CompositeRequest
from contracts.quality_checker import QCRequest, QCResult
from src.application.compile_avatar import AvatarCompiler
from src.application.render_video.audio_probe import _probe_audio_duration
from src.application.render_video.face_region import (
    FACE_REGION_RESOLUTION,
    extract_face_roi,
    mux_audio,
)
from src.domain.body_template import BodyTemplate, load_body_template
from src.domain.types import (
    AvatarIdentityHandle,
    IdentityId,
    IdentitySpec,
    RenderChunkRequest,
)
from src.core.logging import get_logger
from src.quality.exceptions import EncodingError
from providers.compositing.opencv_face.compositor import OpenCVFaceCompositor
from src.quality.video_quality import VideoQualityChecker
from src.storage.avatar_packs import AvatarPackRepository
from src.motion.benchmark import benchmark_pose_track
from src.motion.face_motion_timeline import FaceMotionTimeline
from src.motion.pose_graph import PoseGraphTrack

LOG = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class RenderCachedAvatarResult:
    """Final outcome of :func:`render_cached_avatar`.

    ``status`` carries the QC verdict (``COMPLETED`` or one of the
    ``FAILED_QC_*`` codes). The five cache / mode flags are persisted
    to ``metrics.json`` next to the final mp4 so the bench script can
    compute cost ratios without re-inspecting the engine. ``gpu_seconds``
    is the *real* UNet+VAE measurement from the engine, not a stub.
    """

    status: str
    avatar_id: str
    gesture_id: str
    face_roi_path: Path
    face_lipsynced_path: Path
    composited_path: Path
    final_path: Optional[Path]
    body_dir: Path
    output_seconds: float
    wall_seconds: float
    gpu_seconds: float
    face_resolution: tuple[int, int]
    batch_size: int
    body_cache_hit: bool
    identity_cache_hit: bool
    model_warm: bool
    face_region_only: bool
    qc_result: QCResult
    metrics: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Identity resolution helper
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_identity(
    *,
    avatar_id: str,
    identity_id: Optional[str],
    source_image: Optional[Path],
    pack_repo: Optional[AvatarPackRepository],
    engine: AvatarEngine,
    pack_root: Path,
    identity_pack_path: Optional[Path],
) -> tuple[AvatarIdentityHandle, bool, bool]:
    """Return ``(handle, identity_cache_hit, model_warm)``.

    Resolution order:

    1. ``identity_pack_path`` — explicit pre-built pack on disk.
    2. ``pack_repo.get(identity_id)`` — repository cache hit.
    3. ``AvatarCompiler.compile(source_image)`` — fresh compile.
    """
    if identity_pack_path is not None and identity_pack_path.is_file():
        LOG.debug("Identity pack supplied at %s", identity_pack_path)
        return (
            AvatarIdentityHandle(
                identity_id=IdentityId(identity_id or identity_pack_path.stem),
                pack_path=identity_pack_path,
                pack_digest="sha256:explicit",
                prepared_at=datetime.now(timezone.utc),
            ),
            True,
            # "model_warm" means the engine is actually IDLE and ready to
            # render — not merely loaded_clocks_started. A failed real-mode
            # ``load()`` set ``_loaded_at`` then transitioned to DEGRADED,
            # so we gate on state instead of uptime to avoid lying about
            # cache state to the operator.
            engine.health().state == EngineState.IDLE,
        )

    if identity_id and pack_repo is not None:
        existing = pack_repo.get(IdentityId(identity_id))
        if existing is not None:
            LOG.debug("Identity cache hit at %s", existing.archive_path)
            return (
                AvatarIdentityHandle(
                    identity_id=IdentityId(identity_id),
                    pack_path=existing.archive_path,
                    pack_digest=existing.digest(),
                    prepared_at=existing.manifest.created_at,
                ),
                True,
                engine.health().state == EngineState.IDLE,
            )

    if source_image is None:
        raise RuntimeError(
            "render_cached_avatar: no identity_pack_path, no pack_repo "
            "cache hit, no source_image — cannot compile identity. "
            "Provide source_image (or a pre-built pack) and try again."
        )

    spec = IdentitySpec(
        source_image=source_image,
        display_name=avatar_id,
        language_hint="",
    )
    compiler = AvatarCompiler(engine=engine, pack_root=pack_root)
    handle = compiler.compile(spec)
    # The compiler already wrote the pack to ``{pack_root}/{id}.tar``.
    # The next invocation that resolves through ``pack_repo.get(id)`` will
    # re-load it from disk via :func:`AvatarPackRepository.get`, so we do
    # NOT copy-then-save here (avoids a useless shutil.copy2 and stale
    # manifest round-trip). See ``AvatarPackRepository`` for the disk
    # canonical contract.
    LOG.debug("Identity compiled fresh: %s", handle.identity_id)
    return handle, False, engine.health().state == EngineState.IDLE


# ─────────────────────────────────────────────────────────────────────────────
# Main use case
# ─────────────────────────────────────────────────────────────────────────────


def render_cached_avatar(
    avatar_id: str,
    gesture_id: str,
    identity_id: str,
    audio_path: Path,
    output_path: Path,
    *,
    engine: AvatarEngine,
    source_image: Optional[Path] = None,
    identity_pack_path: Optional[Path] = None,
    pack_repo: Optional[AvatarPackRepository] = None,
    pack_root: Optional[Path] = None,
    capture_dir: Optional[Path] = None,
    fps: int = 25,
    body_template_loader: Callable[..., BodyTemplate] = load_body_template,
    body_templates_dir: Path | str = "body_templates",
    motion_track_path: Optional[Path] = None,
    face_motion_timeline_path: Optional[Path] = None,
    debug: bool = False,
) -> RenderCachedAvatarResult:
    """Run the :class:`BodyTemplate` + face-region + MuseTalk pipeline.

    Args:
        avatar_id: Avatar identity used both as the body-template key
            and the pack directory label.
        gesture_id: Gesture key (``explain_both``, ``idle`` …).
        identity_id: Cache key for the Avatar Pack.
        audio_path: Source WAV / mp3 driving the lip-sync.
        output_path: Destination mp4 (with audio muxed).
        engine: Adapter implementation to call.
        source_image: Optional source PNG used to compile a fresh pack
            when the cache misses.
        identity_pack_path: Optional explicit pack path, short-circuits
            compilation.
        pack_repo: Optional repository for cross-job cache reuse.
        pack_root: Where the compiler writes fresh packs.
        capture_dir: Runtime capture root (defaults to settings.capture_dir).
        fps: Output framerate.
        body_template_loader: Injectable for tests (defaults to
            :func:`load_body_template`).
        body_templates_dir: Base directory for body templates.
        motion_track_path: Optional hand/body motion asset. If supplied,
            the motion summary is included in metrics and can auto-resolve
            ``gesture_id`` when set to ``"auto"``.
        face_motion_timeline_path: Optional hand-free facial timeline JSON
            path. When provided, it is surfaced in the metrics so the
            downstream render/logging flow can stay aligned with the new
            face-only motion layer.
        debug: When True, the compositor writes its debug previews.

    Returns:
        :class:`RenderCachedAvatarResult`.

    Raises:
        FileNotFoundError: body-template or audio assets are missing.
        RuntimeError: the engine raised in real mode (fail-closed).
    """
    settings = engine.settings
    capture_root = capture_dir or settings.capture_dir
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Body template — fail-closed validation ──────────────────────────
    motion_track: PoseGraphTrack | None = None
    motion_summary: dict[str, object] | None = None
    face_motion_summary: dict[str, object] | None = None
    effective_gesture_id = gesture_id
    if motion_track_path is not None and motion_track_path.is_file():
        motion_track = PoseGraphTrack.from_npz(motion_track_path)
        motion_summary = {
            "frames": motion_track.frames,
            "fps": motion_track.fps,
            "active_segment_bounds": motion_track.active_segment_bounds(),
            "unique_states": sorted({state for state in motion_track.pose_state if state and state != "neutral_desk"}),
        }
        if gesture_id == "auto":
            effective_gesture_id = _gesture_from_pose_track(motion_track)
    if face_motion_timeline_path is not None and face_motion_timeline_path.is_file():
        try:
            face_timeline = FaceMotionTimeline.from_dict(
                json.loads(face_motion_timeline_path.read_text(encoding="utf-8"))
            )
            face_motion_summary = {
                "duration": face_timeline.duration,
                "fps": face_timeline.fps,
                "segment_count": len(face_timeline.segments),
                "motion_ids": [seg.motion_id for seg in face_timeline.segments],
                "families": sorted({seg.family for seg in face_timeline.segments}),
            }
        except Exception:
            face_motion_summary = {"path": str(face_motion_timeline_path)}

    body = body_template_loader(avatar_id, effective_gesture_id, base_dir=body_templates_dir)
    body_cache_hit = True
    LOG.info(
        "render_cached_avatar: avatar_id=%s gesture_id=%s body_dir=%s",
        avatar_id, effective_gesture_id, body.body_video.parent,
    )

    # ── 2. Identity resolution (cache, explicit pack, or compile) ───────────
    handle, identity_cache_hit, model_warm = _resolve_identity(
        avatar_id=avatar_id,
        identity_id=identity_id,
        source_image=source_image,
        pack_repo=pack_repo,
        engine=engine,
        pack_root=(pack_root or settings.pack_dir),
        identity_pack_path=identity_pack_path,
    )

    # ── 3. Face-ROI extraction (deterministic 256×256 from body bbox) ───────
    job_id = f"cached-{uuid.uuid4().hex[:12]}"
    runtime_root = capture_root / job_id
    runtime_root.mkdir(parents=True, exist_ok=True)
    debug_dir = (runtime_root / "debug") if debug else None

    face_roi_path = runtime_root / "face_roi.mp4"
    extract_face_roi(
        body.body_video,
        body.face_transforms,
        face_roi_path,
        debug_dir=debug_dir,
        target_size=FACE_REGION_RESOLUTION[0],
    )

    # ── 4. MuseTalk face-region-only inference (single chunk) ──────────────
    audio_duration_seconds = _probe_audio_duration(audio_path)
    chunk_request = RenderChunkRequest(
        job_id=job_id,
        audio_window=(0.0, audio_duration_seconds or 4.0),
        audio_path=audio_path,
        fps=fps,
        resolution=FACE_REGION_RESOLUTION,
        chunk_index=0,
        overlap_seconds=0.0,
        face_region_only=True,
        face_motion_timeline_path=face_motion_timeline_path,
    )
    t0 = time.perf_counter()
    chunk_result = engine.render_chunk(chunk_request, handle)
    wall_seconds = time.perf_counter() - t0
    face_lipsynced_path = chunk_result.output_path

    # ── 5. OpenCVFaceCompositor: paste-back + colour match + feathers ───────
    composited_path = runtime_root / "composited.mp4"
    compositor = OpenCVFaceCompositor()
    composite_result = compositor.composite(
        CompositeRequest(
            body_video=body.body_video,
            generated_face_video=face_lipsynced_path,
            face_mask_video=body.face_mask,
            neck_mask_video=body.neck_mask,
            face_transforms=body.face_transforms,
            output_path=composited_path,
            debug=debug,
        )
    )

    # ── 6. Audio mux ───────────────────────────────────────────────────────
    final_path = runtime_root / "final.mp4"
    try:
        mux_audio(composited_path, audio_path, final_path)
    except Exception as exc:
        raise EncodingError(f"Audio mux failed: {exc}") from exc

    if not final_path.is_file():
        raise EncodingError(f"Audio mux succeeded but output file is missing: {final_path}")

    qc_video_path = final_path

    # ── 7. Six-point QC gate ───────────────────────────────────────────────
    checker = VideoQualityChecker()
    expected_frames = composite_result.frames_processed
    qc_result = checker.check_quality(
        QCRequest(
            video_path=qc_video_path,
            audio_path=audio_path,
            expected_frames=expected_frames,
        )
    )
    if not qc_result.passed:
        # Surface the failure; QC status still recorded in the result.
        LOG.error(
            "render_cached_avatar: QC rejected (%s) green=%.4f black=%.4f "
            "duration_delta=%.1fms",
            qc_result.status, qc_result.debug_green_ratio,
            qc_result.black_frame_ratio, qc_result.duration_delta_ms,
        )
    else:
        LOG.info(
            "render_cached_avatar: QC passed for avatar_id=%s", avatar_id,
        )

    metrics = {
        "status": qc_result.status,
        "avatar_id": avatar_id,
        "gesture_id": effective_gesture_id,
        "identity_id": identity_id,
        "output_seconds": round(chunk_result.duration_seconds, 3),
        "wall_seconds": round(wall_seconds, 3),
        "gpu_seconds": round(chunk_result.gpu_seconds, 4),
        "gpu_seconds_per_output_minute": round(
            chunk_result.gpu_seconds * 60.0 / max(chunk_result.duration_seconds, 1e-3), 4
        ),
        "face_resolution": list(FACE_REGION_RESOLUTION),
        "batch_size": getattr(engine, "render_batch_size", 8),
        "body_cache_hit": body_cache_hit,
        "identity_cache_hit": identity_cache_hit,
        "model_warm": model_warm,
        "face_region_only": True,
        "motion_track_path": str(motion_track_path) if motion_track_path else "",
        "face_motion_timeline_path": str(face_motion_timeline_path)
        if face_motion_timeline_path
        else "",
        "motion_track": motion_summary or {},
        "face_motion_timeline": face_motion_summary or {},
        "qc": {
            "debug_green_ratio": qc_result.debug_green_ratio,
            "black_frame_ratio": qc_result.black_frame_ratio,
            "duration_delta_ms": qc_result.duration_delta_ms,
            "frames_expected": qc_result.frames_expected,
            "frames_actual": qc_result.frames_actual,
        },
    }
    if motion_track is not None:
        motion_benchmark = benchmark_pose_track(motion_track)
        metrics["motion_benchmark"] = {
            "gesture_variety": motion_benchmark.gesture_variety,
            "presence_score": motion_benchmark.presence_score,
            "motion_energy": motion_benchmark.motion_energy,
            "transition_density": motion_benchmark.transition_density,
            "steady_pose_ratio": motion_benchmark.steady_pose_ratio,
        }

    result = RenderCachedAvatarResult(
        status=qc_result.status,
        avatar_id=avatar_id,
        gesture_id=effective_gesture_id,
        face_roi_path=face_roi_path,
        face_lipsynced_path=face_lipsynced_path,
        composited_path=composited_path,
        final_path=final_path,
        body_dir=body.body_video.parent,
        output_seconds=chunk_result.duration_seconds,
        wall_seconds=wall_seconds,
        gpu_seconds=chunk_result.gpu_seconds,
        face_resolution=FACE_REGION_RESOLUTION,
        batch_size=getattr(engine, "render_batch_size", 8),
        body_cache_hit=body_cache_hit,
        identity_cache_hit=identity_cache_hit,
        model_warm=model_warm,
        face_region_only=True,
        qc_result=qc_result,
        metrics=metrics,
    )
    # Write metrics/result.json next to the runtime artifacts (capture_root/<job_id>)
    # so the bench script can find them at the canonical location the
    # tooling section of the plan calls out:
    #   captures/cost_test_001/runtime/{face_roi,face_lipsynced,composited,final}.mp4
    #                                     + result.json + metrics.json
    _write_metrics(capture_root / job_id, metrics, qc_result, final_path)
    return result


def _gesture_from_pose_track(track: PoseGraphTrack) -> str:
    states = list(track.pose_state)
    if any(state == "both_hands_open" for state in states):
        return "explain_both"
    if any(state == "right_hand_up" for state in states):
        return "explain_right"
    if any(state == "left_hand_up" for state in states):
        return "explain_left"
    if any(state == "right_hand_rising" for state in states):
        return "point_right"
    if any(state == "left_hand_rising" for state in states):
        return "point_left"
    return "idle_small"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _write_metrics(
    metrics_dir: Path, metrics: dict, qc_result: QCResult, final_path: Optional[Path],
) -> None:
    """Persist ``metrics.json`` alongside ``result.json`` for benchmark use.

    The bench script reads these two files to compute cost ratios.
    ``result.json`` mirrors the headline status the worker writes;
    ``metrics.json`` carries the full block schema (gates, batch size,
    cache hits) the plan calls "Block 1 metrics".
    """
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    result_path = metrics_dir / "result.json"
    result_payload = {
        "status": qc_result.status,
        "passed": qc_result.passed,
        "errors": list(qc_result.errors),
        "warnings": list(qc_result.warnings),
        "final_path": str(final_path) if final_path else None,
    }
    with result_path.open("w", encoding="utf-8") as fh:
        json.dump(result_payload, fh, indent=2)


__all__ = [
    "RenderCachedAvatarResult",
    "render_cached_avatar",
] 
