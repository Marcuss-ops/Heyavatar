"""Tests for ``src.application.run_cached_avatar_from_text``.

Verifies the text-driven pipeline integration: a timeline plan drives
per-segment renders that are stitched into a single mp4 by the
encoding worker.

Strategy
--------
All tests monkey-patch the two places that would invoke ffmpeg /
write real mp4s:

- :func:`src.application.render_cached_avatar.render_cached_avatar` —
  replaced by a stub that returns a :class:`RenderCachedAvatarResult`
  with a fake final_path, so the orchestrator never touches cv2,
  numpy, or a real engine.
- :class:`src.application.run_cached_avatar_from_text.EncodingWorker` —
  replaced by a stub that just writes a 1-byte final mp4, so the
  orchestrator never invokes ffmpeg.

This keeps the suite hermetic on a lean dev box without numpy/cv2
and without ffmpeg-driven concat logic. Real-mode integration is
covered by ``tests/smoke/test_real_gpu/`` which DOES exercise the
real pipeline (and rightly skips on boxes without CUDA).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pytest

from contracts.avatar_engine import AvatarEngine, EngineHealth, EngineState
from src.application.render_cached_avatar import RenderCachedAvatarResult
from src.application.run_cached_avatar_from_text import (
    run_cached_avatar_from_text,
)
from src.domain.enums import EngineId
from src.domain.types import (
    AvatarIdentityHandle,
    IdentityId,
    RenderChunkRequest,
    RenderChunkResult,
)


RECEIVED_FACE_MOTION_TIMELINES: List[Path | None] = []


# ─────────────────────────────────────────────────────────────────────────────
# Test scaffolds
# ─────────────────────────────────────────────────────────────────────────────


class _StubEngine(AvatarEngine):
    """Hermetic engine that records every render call.

    Not actually invoked by the test path — the orchestrator
    receives this engine instance for ownership but never reaches
    into ``render_chunk`` because the render_cached_avatar stub
    short-circuits before that.
    """

    engine_id = EngineId.MUSE_TALK

    def __init__(self) -> None:
        super().__init__()
        self._loaded = False
        self.render_calls: List[RenderChunkRequest] = []

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def prepare_identity(self, source_image: Path) -> AvatarIdentityHandle:
        return AvatarIdentityHandle(
            identity_id=IdentityId("stub"),
            pack_path=source_image,
            pack_digest="sha256:stub",
            prepared_at=None,  # type: ignore[arg-type]
        )

    def render_chunk(
        self,
        request: RenderChunkRequest,
        identity: AvatarIdentityHandle,
    ) -> RenderChunkResult:
        if not self._loaded:
            raise RuntimeError("engine not loaded")
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        request.output_path.write_bytes(b"\x00")
        self.render_calls.append(request)
        return RenderChunkResult(
            chunk_index=request.chunk_index,
            output_path=request.output_path,
            duration_seconds=max(0.001, request.audio_window[1] - request.audio_window[0]),
            frames_rendered=1,
            gpu_seconds=0.0,
            engine_id=self.engine_id,
        )

    def health(self) -> EngineHealth:
        return EngineHealth(
            engine_id=self.engine_id,
            state=EngineState.IDLE if self._loaded else EngineState.UNLOADED,
        )


@dataclass(slots=True)
class _StubSettings:
    """Settings-like stand-in. EncodingWorker only reads ``output_dir``
    from itself, so a minimal shape is enough for the orchestrator."""

    pack_dir: Path
    capture_dir: Path
    motion_style: str = "natural"


def _make_engine_with_settings() -> _StubEngine:
    """Return a stub engine with a usable ``.settings`` attribute."""
    engine = _StubEngine()
    engine.settings = _StubSettings(  # type: ignore[attr-defined]
        pack_dir=Path("./avatar_packs"),
        capture_dir=Path("./captures"),
        motion_style="natural",
    )
    return engine


# ─────────────────────────────────────────────────────────────────────────────
# Stub render_cached_avatar (avoids real engine + cv2 path)
# ─────────────────────────────────────────────────────────────────────────────


def _stub_render_cached_avatar(
    avatar_id: str,
    gesture_id: str,
    identity_id: str,
    audio_path: Path,
    output_path: Path,
    *,
    engine: AvatarEngine,
    source_image: Path | None = None,
    identity_pack_path: Path | None = None,
    pack_repo=None,
    pack_root: Path | None = None,
    capture_dir: Path | None = None,
    fps: int = 25,
    body_templates_dir: str = "body_templates",
    face_motion_timeline_path: Path | None = None,
    debug: bool = False,
    **_: object,
) -> RenderCachedAvatarResult:
    """Replacement for :func:`render_cached_avatar`.

    Writes a 1-byte output and returns the silhouette result the
    orchestrator expects. Pure stdlib.
    """
    seg_dir = capture_dir or audio_path.parent
    RECEIVED_FACE_MOTION_TIMELINES.append(face_motion_timeline_path)
    final_path = seg_dir / "final.mp4"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_bytes(b"\x00")
    return RenderCachedAvatarResult(
        status="COMPLETED",
        avatar_id=avatar_id,
        gesture_id=gesture_id,
        face_roi_path=seg_dir / "face_roi.mp4",
        face_lipsynced_path=seg_dir / "face_lipsynced.mp4",
        composited_path=seg_dir / "composited.mp4",
        final_path=final_path,
        body_dir=seg_dir,
        output_seconds=2.0,
        wall_seconds=1.0,
        gpu_seconds=0.0,
        face_resolution=(256, 256),
        batch_size=8,
        body_cache_hit=True,
        identity_cache_hit=True,
        model_warm=True,
        face_region_only=True,
        qc_result=_make_stub_qc_result(),
    )


def _make_stub_qc_result() -> "object":
    """Build a QCResult-shaped namespace the orchestrator never deeply
    inspects — keeps the test free of the real QualityResult dataclass
    + its numpy/cv2 dependencies."""

    class _QC:
        status = "COMPLETED"
        passed = True
        green_ratio = 0.0
        black_frame_ratio = 0.0
        duration_delta_ms = 0.0
        frames_expected = 50
        frames_actual = 50
        errors: List[str] = []
        warnings: List[str] = []

    # Build a real QCResult so the type contract holds. If the
    # import resolves we get the real dataclass; otherwise stub.
    try:
        from contracts.quality_checker import QCResult  # type: ignore

        return QCResult(
            status="COMPLETED",
            passed=True,
            debug_green_ratio=0.0,
            black_frame_ratio=0.0,
            duration_delta_ms=0.0,
            frames_expected=50,
            frames_actual=50,
            errors=[],
            warnings=[],
        )
    except Exception:
        return _QC()


def _stub_encoding_worker_factory(*_args, **_kwargs) -> "object":
    """Replacement for :class:`EncodingWorker`.

    Comes back with a stub that writes a 1-byte final mp4 and
    returns it (no ffmpeg). Mirrors ``EncodingWorker.encode()``
    enough for the orchestrator's path."""
    settings = _args[0] if _args else _kwargs.get("settings", _StubSettings(
        pack_dir=Path("./avatar_packs"), capture_dir=Path("./captures")
    ))

    class _StubWorker:
        def __init__(self, _settings) -> None:
            self.settings = _settings
            self.output_dir = _kwargs.get("output_dir", Path("./captures"))
            self.overlap_seconds = _kwargs.get("overlap_seconds", 0.0)
            self.last_calls: List[Tuple[str, Path]] = []

        def encode(self, job_id: str, manifest_path: Path, *, audio_path=None, codec="h264"):
            out = self.output_dir / f"{job_id}.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"\x00")
            self.last_calls.append((job_id, manifest_path))
            return out

    return _StubWorker(settings)


def _stub_audio_slicer(_: Path, start: float, end: float, dest: Path) -> Path:
    """Audio slicing stub — writes a 1-byte filename-encoded file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"\x00")
    return dest


# ─────────────────────────────────────────────────────────────────────────────
# Body-template writers
# ─────────────────────────────────────────────────────────────────────────────


def _write_synthetic_body_template(
    base_dir: Path,
    avatar_id: str,
    gesture_id: str,
) -> None:
    """Materialise body-template files for :func:`load_body_template`."""
    pack = base_dir / avatar_id / "body_cache" / gesture_id
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "body.mp4").write_bytes(b"\x00")
    (pack / "face_mask.mp4").write_bytes(b"\x00")
    (pack / "neck_mask.mp4").write_bytes(b"\x00")
    # Use stdlib-only Python to write the npz; nil numpy dependency
    # on the test path:
    from struct import pack as _pack
    import zipfile
    npz = pack / "face_transforms.npz"
    with zipfile.ZipFile(npz, "w") as zf:
        # header + 1 zero-byte array — size is irrelevant, file just
        # needs to exist for :func:`load_body_template`'s ``is_file``.
        zf.writestr("_header", b"")
    (pack / "metadata.json").write_text(
        json.dumps(
            {
                "avatar_id": avatar_id,
                "gesture_id": gesture_id,
                "fps": 25,
                "frames": 75,
                "width": 64,
                "height": 64,
                "status": "ok",
            }
        ),
        encoding="utf-8",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pytest fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def stub_orchestrator(monkeypatch):
    """Replace render_cached_avatar + EncodingWorker with hermetic stubs.

    Tests get the ``engine`` so they can assert load/unload semantics.
    Audio slicer is also stubbed by default; tests can override.

    Identity preflight is ALSO stubbed by default — most tests are
    about timeline / fallback / audio-slicing / metrics and don't
    care about identity resolution. The dedicated
    ``test_no_identity_inputs_raises_runtime_error`` deliberately
    drops this fixture to exercise the real ``_require_identity_inputs``.
    """
    import src.application.run_cached_avatar_from_text as lib

    monkeypatch.setattr(lib, "render_cached_avatar", _stub_render_cached_avatar)
    monkeypatch.setattr(lib, "EncodingWorker", _stub_encoding_worker_factory)
    monkeypatch.setattr(
        lib,
        "_require_identity_inputs",
        lambda **_: Path("./fake_identity_source.png"),
    )

    engine = _make_engine_with_settings()
    engine.load()
    RECEIVED_FACE_MOTION_TIMELINES.clear()
    return engine


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_timeline_mode_renders_one_chunk_per_segment(
    stub_orchestrator, tmp_path: Path
) -> None:
    base = tmp_path / "body_templates"
    _write_synthetic_body_template(base, avatar_id="alice", gesture_id="idle_small")
    _write_synthetic_body_template(base, avatar_id="alice", gesture_id="explain_both")
    _write_synthetic_body_template(base, avatar_id="alice", gesture_id="count_three")

    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"\x00")
    output = tmp_path / "out.mp4"

    result = run_cached_avatar_from_text(
        text="Ciao a tutti, oggi vediamo tre differenze. Ti spiego come fare.",
        audio_path=audio,
        output_path=output,
        avatar_id="alice",
        engine=stub_orchestrator,
        fps=25,
        mode="timeline",
        body_templates_dir=base,
        capture_root=tmp_path / "captures",
        audio_slicer=_stub_audio_slicer,
    )

    # The Italian-language planner picks comparison/question/conclusion
    # gestures for this script — none of those have body templates in
    # the test fixture, so they fall back to idle_small. Accept both
    # COMPLETED (no fallbacks needed) and COMPLETED_WITH_FALLBACKS.
    assert result.status in {"COMPLETED", "COMPLETED_WITH_FALLBACKS"}, (
        f"unexpected status {result.status!r}; skipped={result.skipped_gestures}"
    )
    # If we landed on COMPLETED_WITH_FALLBACKS, the fallback path
    # MUST have actually substituted something — otherwise the test
    # would silently pass without exercising the substitution logic.
    if result.status == "COMPLETED_WITH_FALLBACKS":
        assert result.skipped_gestures, (
            "expected at least one substituted gesture when status reports fallback"
        )
    assert result.mode == "timeline"
    assert result.segment_count >= 2
    assert result.output_path.is_file()
    assert result.chunk_manifest_path.is_file()
    assert RECEIVED_FACE_MOTION_TIMELINES
    assert RECEIVED_FACE_MOTION_TIMELINES[0] is not None
    metrics_path = tmp_path / "captures" / result.job_id / "metrics.json"
    assert metrics_path.is_file()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["mode"] == "timeline"
    assert metrics["segment_count"] == result.segment_count


def test_dominant_mode_collapses_timeline_to_single_segment(
    stub_orchestrator, tmp_path: Path
) -> None:
    base = tmp_path / "body_templates"
    # dominant mode dispatches into a single full-clip render against
    # the chosen gesture. Preflight still validates that the *fallback*
    # (idle_small) exists on disk so a wholly missing template set is
    # surfaced as a one-line RuntimeError instead of a per-segment
    # miss loop.
    _write_synthetic_body_template(base, avatar_id="alice", gesture_id="idle_small")
    _write_synthetic_body_template(base, avatar_id="alice", gesture_id="explain_both")
    _write_synthetic_body_template(base, avatar_id="alice", gesture_id="count_three")

    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"\x00")

    result = run_cached_avatar_from_text(
        text="Ciao, oggi vediamo tre differenze importanti tra le opzioni.",
        audio_path=audio,
        output_path=tmp_path / "out.mp4",
        avatar_id="alice",
        engine=stub_orchestrator,
        fps=25,
        mode="dominant",
        body_templates_dir=base,
        capture_root=tmp_path / "captures",
        audio_slicer=_stub_audio_slicer,
    )

    assert result.status == "COMPLETED"
    assert result.mode == "dominant"
    assert result.segment_count == 1


def test_missing_body_template_falls_back_to_idle_small(
    stub_orchestrator, tmp_path: Path
) -> None:
    base = tmp_path / "body_templates"
    _write_synthetic_body_template(base, avatar_id="alice", gesture_id="idle_small")
    _write_synthetic_body_template(base, avatar_id="alice", gesture_id="explain_both")
    _write_synthetic_body_template(base, avatar_id="alice", gesture_id="count_three")

    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"\x00")

    result = run_cached_avatar_from_text(
        text="benvenuti, oggi vediamo tre differenze molto importanti",
        audio_path=audio,
        output_path=tmp_path / "out.mp4",
        avatar_id="alice",
        engine=stub_orchestrator,
        fps=25,
        mode="timeline",
        dominant_fallback_gesture="idle_small",
        body_templates_dir=base,
        capture_root=tmp_path / "captures",
        audio_slicer=_stub_audio_slicer,
    )

    assert result.status in {"COMPLETED", "COMPLETED_WITH_FALLBACKS"}
    assert result.output_path.is_file()


def test_audio_slicer_called_per_segment_with_segment_bounds(
    stub_orchestrator, tmp_path: Path
) -> None:
    base = tmp_path / "body_templates"
    _write_synthetic_body_template(base, avatar_id="alice", gesture_id="idle_small")
    _write_synthetic_body_template(base, avatar_id="alice", gesture_id="explain_both")
    _write_synthetic_body_template(base, avatar_id="alice", gesture_id="count_three")

    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"\x00")

    captured_calls: List[Tuple[float, float]] = []

    def _capturing_slicer(_input: Path, start: float, end: float, dest: Path) -> Path:
        captured_calls.append((start, end))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00")
        return dest

    run_cached_avatar_from_text(
        text="Ciao a tutti, oggi parliamo di tre differenze",
        audio_path=audio,
        output_path=tmp_path / "out.mp4",
        avatar_id="alice",
        engine=stub_orchestrator,
        fps=25,
        mode="timeline",
        body_templates_dir=base,
        capture_root=tmp_path / "captures",
        audio_slicer=_capturing_slicer,
    )

    assert captured_calls, "slicer should have been called at least once"
    sorted_calls = sorted(captured_calls)
    for prev, nxt in zip(sorted_calls, sorted_calls[1:]):
        assert nxt[0] >= prev[1] - 1e-3


def test_invalid_mode_raises_value_error(stub_orchestrator, tmp_path: Path) -> None:
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"\x00")

    with pytest.raises(ValueError, match="mode must be"):
        run_cached_avatar_from_text(
            text="qualsiasi cosa",
            audio_path=audio,
            output_path=tmp_path / "out.mp4",
            avatar_id="alice",
            engine=stub_orchestrator,
            fps=25,
            mode="bogus_mode",
            body_templates_dir=tmp_path / "no_templates",
            capture_root=tmp_path / "captures",
            audio_slicer=_stub_audio_slicer,
        )


def test_result_metrics_carry_timeline_plan(stub_orchestrator, tmp_path: Path) -> None:
    base = tmp_path / "body_templates"
    for gid in ("idle_small", "explain_both", "count_three"):
        _write_synthetic_body_template(base, avatar_id="bob", gesture_id=gid)

    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"\x00")

    result = run_cached_avatar_from_text(
        text="Ciao a tutti, oggi vediamo tre differenze",
        audio_path=audio,
        output_path=tmp_path / "out.mp4",
        avatar_id="bob",
        engine=stub_orchestrator,
        fps=25,
        mode="timeline",
        body_templates_dir=base,
        capture_root=tmp_path / "captures",
        audio_slicer=_stub_audio_slicer,
    )

    timeline_gestures = result.metrics["timeline_gestures"]
    assert len(timeline_gestures) == len(result.timeline.segments)
    assert set(timeline_gestures) == {s.gesture_id for s in result.timeline.segments}


def test_no_body_template_for_fallback_raises_at_entry(stub_orchestrator, tmp_path: Path) -> None:
    """If even the fallback gesture has no template, the library fails
    fast with a path-aware error rather than looping into the same
    miss per-segment."""
    base = tmp_path / "body_templates"
    # Seed nothing — neither the planned gestures nor idle_small.
    _write_synthetic_body_template(base, avatar_id="alice", gesture_id="explain_both")
    _write_synthetic_body_template(base, avatar_id="alice", gesture_id="count_three")
    # idle_small intentionally absent.

    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"\x00")

    with pytest.raises(RuntimeError, match="no body templates on disk"):
        run_cached_avatar_from_text(
            text="Ciao, oggi vediamo tre differenze",
            audio_path=audio,
            output_path=tmp_path / "out.mp4",
            avatar_id="alice",
            engine=stub_orchestrator,
            fps=25,
            mode="timeline",
            dominant_fallback_gesture="idle_small",
            body_templates_dir=base,
            capture_root=tmp_path / "captures",
            audio_slicer=_stub_audio_slicer,
        )


def test_no_identity_inputs_raises_runtime_error(tmp_path: Path) -> None:
    """Without ``--source-image`` / ``--identity-pack`` / ``pack_repo``
    the library function raises with a clear message instead of
    silently fabricating ``/dev/null``.

    Note: deliberately DOES NOT use ``stub_orchestrator`` so it
    exercises the real ``_require_identity_inputs`` preflight.
    """
    base = tmp_path / "body_templates"
    _write_synthetic_body_template(base, avatar_id="alice", gesture_id="idle_small")

    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"\x00")

    # Fresh engine not touched by the test fixture.
    engine = _make_engine_with_settings()
    engine.load()

    with pytest.raises(RuntimeError, match="no identity_pack_path"):
        run_cached_avatar_from_text(
            text="qualsiasi cosa",
            audio_path=audio,
            output_path=tmp_path / "out.mp4",
            avatar_id="alice",
            engine=stub_orchestrator,
            fps=25,
            mode="timeline",
            body_templates_dir=base,
            capture_root=tmp_path / "captures",
            audio_slicer=_stub_audio_slicer,
        )


def test_engine_id_resolution_accepts_canonical_names() -> None:
    """The CLI engine-name resolver accepts the dash-separated
    canonical names from ``registry/models.yaml``."""
    import sys
    sys.path.insert(0, ".")
    from tools.run_cached_avatar import _resolve_engine_id

    from src.domain.enums import EngineId

    assert _resolve_engine_id("musetalk-v1") is EngineId.MUSE_TALK
    assert _resolve_engine_id("liveportrait-human-v1") is EngineId.LIVE_PORTRAIT
    assert _resolve_engine_id("echomimic-v1") is EngineId.ECHO_MIMIC
    assert _resolve_engine_id(None) is EngineId.MUSE_TALK
    with pytest.raises(ValueError, match="unknown engine"):
        _resolve_engine_id("totally-mock")


def test_run_cached_avatar_from_text_temp_overrides_motion_style(
    stub_orchestrator, tmp_path: Path
) -> None:
    base = tmp_path / "body_templates"
    _write_synthetic_body_template(base, avatar_id="alice", gesture_id="idle_small")
 
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"\x00")
 
    # Set initial motion style
    stub_orchestrator.settings.motion_style = "natural"
 
    seen_styles = {}
 
    # We patch render_cached_avatar to capture the motion style during execution
    import src.application.run_cached_avatar_from_text as lib
    original_render = lib.render_cached_avatar
 
    def _render_and_capture(*args, **kwargs):
        seen_styles["during"] = getattr(kwargs["engine"].settings, "motion_style", None)
        return original_render(*args, **kwargs)
 
    from unittest.mock import patch
    with patch("src.application.run_cached_avatar_from_text.render_cached_avatar", _render_and_capture):
        run_cached_avatar_from_text(
            text="Ciao a tutti",
            audio_path=audio,
            output_path=tmp_path / "out.mp4",
            avatar_id="alice",
            engine=stub_orchestrator,
            fps=25,
            mode="timeline",
            body_templates_dir=base,
            capture_root=tmp_path / "captures",
            audio_slicer=_stub_audio_slicer,
            motion_style="expressive",
        )
 
    assert seen_styles["during"] == "expressive"
    assert stub_orchestrator.settings.motion_style == "natural"
