"""Integration test — ``RenderVideo.run_timeline`` end-to-end.

Exercises the orchestrator's deterministic multi-template pathway
(``docs/REPOSITORY_SLIMMING_PLAN.md`` §6 + §10, Change 4) against a
synthesised ``body_templates`` tree, an 8.0-second silent WAV, and the
canonical reference timeline JSON at ``docs/examples/timeline_three_segment.json``.

The test is the orchestrator's golden signal:

* Synth 3 body templates (75 / 50 / 75 frames @ 25 fps).
* Synth an audio file of EXACTLY 8.0 seconds — the sum of segment
  durations — so the audio-vs-aligned-drift check on
  ``RenderVideo.run_timeline`` is exercised.
* Load ``Timeline.from_json`` from the reference JSON.
* Construct ``AvatarIdentityHandle`` + ``RenderRequest``.
* Monkey-patch ``load_body_template`` so it resolves to the synthesised
  tree rather than the canonical ``body_templates/`` directory.
* Call ``RenderVideo.run_timeline(...)``.
* Assert the returned :class:`AlignedBodyTimeline` matches the
  deterministically derivable facts implicit in the JSON: 200 frames,
  25 fps, 8.0 s, single 64×64 resolution, all 5 canonical files
  written, metadata.json segments breakdown matches the JSON.

If ffprobe is unavailable locally, the audio-vs-aligned-drift check
inside ``run_timeline`` short-circuits (logs a warning, no assertion),
but the rest of the test path is still fully exercised.
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pytest

from src.application.render_video.use_case import RenderVideo
from contracts.avatar_engine import AvatarEngine
from src.domain.body_template import BodyTemplate
from src.domain.enums import EngineId, Tier
from src.domain.timeline import Timeline
from src.domain.types import (
    AvatarIdentityHandle,
    IdentityId,
    IdentitySpec,
    RenderJobId,
    RenderRequest,
    RenderSpec,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLE_JSON = REPO_ROOT / "docs" / "examples" / "timeline_three_segment.json"

# Contract pin: ``Timeline.from_dict`` currently tolerates unknown
# top-level keys (lax mode), so ``_comment`` in the reference JSON
# is silently dropped at load time. If the loader is ever
# tightened to strict mode (rejects unknown keys), this golden-
# signal test will fail at the Step 4a ``_lax_from_dict`` assertion
# below — see ``src/domain/timeline.py::Timeline.from_dict`` for the
# implementation.


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic fixture builders (mirror the unit-test style so this integration
# test stays self-contained; avoids cross-test directory imports).
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
    bbox_margin: int = 8,
) -> Path:
    """Materialise a single on-disk body_template at ``base_dir``.

    Returns the resulting template directory (mirrors the convention
    enforced by ``tools/avatar_assets/precompute_video_template.py``).
    """
    tmpl_dir = base_dir / avatar_id / gesture_id
    tmpl_dir.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame = np.full((height, width, 3), body_colour, dtype=np.uint8)
    fmask_frame = np.full((height, width, 3), fmask_colour, dtype=np.uint8)
    nmask_frame = np.full((height, width, 3), nmask_colour, dtype=np.uint8)

    for name, payload in (
        ("body.mp4", frame),
        ("face_mask.mp4", fmask_frame),
        ("neck_mask.mp4", nmask_frame),
    ):
        writer = cv2.VideoWriter(
            str(tmpl_dir / name), fourcc, fps, (width, height),
        )
        assert writer.isOpened(), f"failed to open writer for {tmpl_dir / name}"
        for _ in range(total_frames):
            writer.write(payload)
        writer.release()

    bbox = np.tile(
        np.array(
            [bbox_margin, bbox_margin, width - bbox_margin, height - bbox_margin],
            dtype=np.float32,
        ),
        (total_frames, 1),
    )
    matrices = np.tile(np.eye(4, dtype=np.float32), (total_frames, 1, 1))
    # Mirror tools/avatar_assets/precompute_video_template.py's
    # npz layout: real precompute writes bbox + matrices + landmarks
    # (+ optional confidence). The synth helper writes landmarks
    # so align_timeline's lmk dtype-normalise code path is
    # exercised end-to-end (without this, the integration test
    # silently skips MED 1's "landmarks dtype drift" invariants).
    landmarks = np.zeros((total_frames, 478, 3), dtype=np.float32)
    landmarks[:, :, 0] = np.linspace(0.1, 0.9, 478, dtype=np.float32)  # x
    landmarks[:, :, 1] = np.linspace(0.2, 0.6, 478, dtype=np.float32)  # y
    landmarks[:, :, 2] = np.full(478, 0.5, dtype=np.float32)            # z
    np.savez_compressed(
        tmpl_dir / "face_transforms.npz",
        bbox=bbox,
        matrices=matrices,
        landmarks=landmarks,
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


def _make_wav(path: Path, duration_s: float, sample_rate: int = 16000) -> None:
    """Write a minimal PCM WAV file (mono silence) of ``duration_s`` seconds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = int(sample_rate * duration_s)
    data_size = n_samples * 2  # 16-bit samples
    with open(path, "wb") as f:
        # RIFF header.
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        # fmt chunk.
        f.write(b"fmt ")
        f.write(struct.pack(
            "<IHHIIHH", 16, 1, 1, sample_rate,
            sample_rate * 2, 2, 16,
        ))
        # data chunk.
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(b"\x00" * data_size)


def _build_synth_loader(
    base_dir: Path,
) -> Callable[[str, str], BodyTemplate]:
    """Return a ``load_body_template``-shaped closure rooted at ``base_dir``."""
    def loader(avatar_id: str, gesture_id: str) -> BodyTemplate:
        template_dir = base_dir / avatar_id / gesture_id
        return BodyTemplate(
            body_video=template_dir / "body.mp4",
            face_mask=template_dir / "face_mask.mp4",
            neck_mask=template_dir / "neck_mask.mp4",
            face_transforms=template_dir / "face_transforms.npz",
            metadata=template_dir / "metadata.json",
        )
    return loader


# ─────────────────────────────────────────────────────────────────────────────
# Integration test
# ─────────────────────────────────────────────────────────────────────────────


class TestRunTimelineIntegration:
    """End-to-end golden-signal test for the orchestrator's timeline path."""

    def test_run_timeline_full_pipeline_matches_reference_json(
        self, tmp_path, monkeypatch,
    ):
        """Synth 3 body templates + 8.0s WAV + run_timeline → golden signal."""
        # 1. Sanity-check the reference JSON actually exists in the repo.
        assert EXAMPLE_JSON.is_file(), (
            f"reference timeline JSON missing at {EXAMPLE_JSON}; tests/"
            f"integration/test_run_timeline_integration.py depends on "
            f"docs/examples/timeline_three_segment.json."
        )

        # 2. Build the synthesised body_templates tree.
        synth_root = tmp_path / "synth_body_templates"
        # The reference timeline JSON uses gesture_ids "idle" and
        # "explain_both" (with idle bookending the timeline) at the
        # canonical profile of 25fps. Mirror that.
        _synth_template(
            synth_root, "alice", "idle",
            width=64, height=64, fps=25.0, total_frames=75,
            body_colour=(110, 110, 110), fmask_colour=(220, 0, 0),
        )
        _synth_template(
            synth_root, "alice", "explain_both",
            width=64, height=64, fps=25.0, total_frames=50,
            body_colour=(90, 90, 140), fmask_colour=(0, 220, 0),
        )
        # Note: the third segment reuses "idle" per the reference JSON.
        # load_body_template will resolve to the same on-disk template
        # twice; that is the canonical deterministic re-use pattern.

        # 3. Synth an 8.0-second silent WAV that matches the timeline.
        audio_path = tmp_path / "speech.wav"
        _make_wav(audio_path, duration_s=8.0)        # 4. Load the canonical Timeline from the reference JSON.
        # 4a. Round-trip the writer path so the lax-from_dict contract
        #     has a concrete audit trail in both directions. Use SUBSET
        #     assertions (not exact-equal dict) so a future legitimate
        #     additive to_dict key (e.g. "interpolation_seconds") is
        #     silently accepted instead of breaking the golden signal.
        _raw_doc = json.loads(EXAMPLE_JSON.read_text())
        assert "_comment" in _raw_doc, (
            f"reference JSON at {EXAMPLE_JSON} must carry the inline "
            f"_comment block; if it was stripped on save the lax "
            f"from_dict contract pin below becomes vacuous."
        )
        _round_trip = Timeline.from_dict(_raw_doc).to_dict()
        # Raw-doc side: _comment is present in the source file.
        # Round-trip side: _comment is silently dropped + fps/segments
        # preserve their canonical 3-segment shape.
        assert "_comment" not in _round_trip, (
            f"Timeline.from_dict(...) regressed: _comment leaked into "
            f"the round-tripped dict ({sorted(_round_trip)}); either "
            f"from_dict became strict or the writer preserves extra "
            f"keys."
        )
        assert _round_trip.get("fps") == 25, (
            f"fps did not survive from_dict/to_dict round-trip; "
            f"got {_round_trip.get('fps')}, expected 25."
        )
        assert len(_round_trip.get("segments", [])) == 3, (
            f"segment count did not survive round-trip; got "
            f"{len(_round_trip.get('segments', []))}, expected 3."
        )
        assert [s.get("gesture_id") for s in _round_trip["segments"]] == [
            "idle", "explain_both", "idle",
        ], (
            f"segment gesture_ids drifted through round-trip; got "
            f"{[s.get('gesture_id') for s in _round_trip['segments']]}, "
            f"expected ['idle', 'explain_both', 'idle']."
        )
        timeline = Timeline.from_json(EXAMPLE_JSON)
        assert timeline.fps == 25
        assert len(timeline.segments) == 3
        assert timeline.total_duration_seconds() == pytest.approx(8.0)
        assert timeline.expected_frames() == 200
        # Mirror of the JSON.
        assert [s.gesture_id for s in timeline.segments] == [
            "idle", "explain_both", "idle",
        ]
        assert [s.duration_seconds for s in timeline.segments] == [
            3.0, 2.0, 3.0,
        ] 


        # 5. Build the orchestrator inputs.
        identity = AvatarIdentityHandle(
            identity_id=IdentityId("alice"),
            pack_path=tmp_path / "alice_pack",
            pack_digest="synth_digest",
            prepared_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        request = RenderRequest(
            job_id=RenderJobId("test_job_e2e"),
            identity_id=IdentityId("alice"),
            identity_spec=IdentitySpec(
                source_image=tmp_path / "alice.png",
                display_name="Alice",
            ),
            render_spec=RenderSpec(
                audio_path=audio_path,
                fps=25,
                target_resolution=(64, 64),
            ),
            tier=Tier.EXPRESS,
        )

        # 6. Wire run_timeline's body_template_loader at our synth root.
        orchestrator = RenderVideo(
            engine=_StubEngine(),
        )
        loader = _build_synth_loader(synth_root)
        # Patch (auto-restored at teardown) the use_case's bound
        # reference to load_body_template so the orchestrator resolves
        # at synth_root instead of the canonical body_templates/
        # directory. Using monkeypatch.setattr (not the raw
        # __globals__ mutation the previous iteration used) prevents
        # test-isolation leaks across the rest of the pytest suite.
        monkeypatch.setattr(
            "src.application.render_video.use_case.load_body_template",
            loader,
        )

        alignment_dir = tmp_path / "aligned"

        # 7. Run the orchestrator end-to-end.
        aligned, audio_duration = orchestrator.run_timeline(
            timeline,
            "alice",
            identity,
            request,
            alignment_dir,
        )

        # 8. Golden-signal assertions on the returned AlignedBodyTimeline.
        assert aligned.total_frames == 200, (
            f"timeline JSON prescribes 200 frames @ 25fps × 8.0s; "
            f"got {aligned.total_frames}"
        )
        assert aligned.fps == 25
        assert aligned.width == 64 and aligned.height == 64
        assert aligned.duration_seconds == pytest.approx(8.0)

        # 9. The aligned 4-file tree + metadata.json must all exist on disk.
        for path in (
            aligned.body_video,
            aligned.face_mask,
            aligned.neck_mask,
            aligned.face_transforms,
            aligned.metadata,
        ):
            assert Path(path).is_file(), f"missing aligned asset: {path}"

        # 10. metadata.json breakdown must mirror the JSON exactly.
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
        assert [s["gesture_id"] for s in meta["segments"]] == [
            "idle", "explain_both", "idle",
        ]
        assert [s["duration_seconds"] for s in meta["segments"]] == [
            3.0, 2.0, 3.0,
        ]
        assert [s["frames"] for s in meta["segments"]] == [75, 50, 75]

        # 11. The aligned npz must contain the canonical 200-frame arrays.
        data = np.load(str(aligned.face_transforms))
        assert data["bbox"].shape == (200, 4)
        assert data["matrices"].shape == (200, 4, 4)
        assert data["bbox"].dtype == np.float32
        assert data["matrices"].dtype == np.float32
        # landmarks invariant (closes MED 3's test-assertion gap):
        # align_timeline's lmk_parts.append + dtype-normalise branch
        # must have fired here, otherwise this golden-signal test
        # has no audit trail for the landmarks invariant.
        assert data["landmarks"].shape == (200, 478, 3), (
            f"landmarks invariant failure: expected (200, 478, 3); "
            f"got {data['landmarks'].shape}. Either "
            f"_synth_template stopped writing landmarks, or "
            f"align_timeline's landmarks concat path regressed."
        )
        assert data["landmarks"].dtype == np.float32, (
            f"landmarks dtype invariant: expected float32 (the "
            f"aligner normalises cross-segment dtype to float32); "
            f"got {data['landmarks'].dtype}"
        )
        # Landmarks content sanity — if align_timeline's landmarks
        # dtype-normalise path regresses into a NaN/inf-producing
        # branch (e.g. cross-segment float64 source + a buggy cast),
        # this assertion catches it where the shape + dtype assertion
        # above would not.
        assert np.all(np.isfinite(data["landmarks"])), (
            "landmarks contain NaN/inf after align_timeline's "
            "dtype-normalise; the synth seeded all-zero / "
            "linspace-only-finite values so this can only be "
            "caused by the aligner regressing into a non-finite "
            "intermediate (or the synth seeding broken values)."
        )
        # Landmarks variability — np.isfinite alone accepts the all-
        # zeros regression (0.0 is finite), so a real-teeth check
        # also asserts the value distribution has spread. The synth
        # seeds np.linspace-driven x and y values per frame, so even
        # the worst-case regression that returns constant 0.0 will
        # trip this check (std == 0).
        assert data["landmarks"].std() > 0, (
            "landmarks after align_timeline's dtype-normalise have "
            "constant value (std == 0); synth seeded arrays of "
            "linspace-driven x/y and constant z=0.5, so this can "
            "only be caused by the synth seeding zeros (real "
            "MediaPipe never produces all-zero landmarks) or the "
            "aligner regressing into a constant-array intermediate."
        )
        # timestamp_ms must be strictly monotonic at 1000/25 = 40ms dt.
        ts = data["timestamp_ms"]
        assert np.all(np.diff(ts) == 40)
        # Full-timeline edges (frame 0 + final frame).
        assert ts[0] == 0
        assert ts[199] == 199 * 40
        # Per-segment boundary assertions — pin the npz cursor against
        # ``metadata.json::segments[].frames = [75, 50, 75]``. Off-by-one
        # at a segment boundary would leave shape + dtype + overall
        # monotonicity intact while silently corrupting downstream
        # frame-indexing consumers that read by segment.
        assert ts[74] == 74 * 40, (
            f"end-of-segment-0 cursor mismatch: ts[74]={ts[74]}, "
            f"expected 74*40={74 * 40} (=3.0s of idle)."
        )
        assert ts[75] == 75 * 40, (
            f"start-of-segment-1 cursor must continue without gap or "
            f"duplicate; ts[75]={ts[75]}, expected 75*40={75 * 40}."
        )
        assert ts[124] == 124 * 40, (
            f"end-of-segment-1 cursor mismatch: ts[124]={ts[124]}, "
            f"expected 124*40={124 * 40} (=5.0s elapsed = end of "
            f"explain_both)."
        )
        assert ts[125] == 125 * 40, (
            f"start-of-segment-2 cursor must continue without gap or "
            f"duplicate; ts[125]={ts[125]}, expected 125*40={125 * 40}."
        )

        # 12. Audio duration vs aligned duration — if ffprobe is available,
        # they must match within the 1-frame tolerance that run_timeline
        # itself enforces; if ffprobe is missing, run_timeline logs a warning
        # and short-circuits — assert the orchestrator didn't crash.
        if audio_duration > 0:
            tolerance = aligned.duration_seconds / aligned.fps  # one frame
            drift = abs(audio_duration - aligned.duration_seconds)
            assert drift <= tolerance, (
                f"audio duration drift {drift:.4f}s exceeds one-frame "
                f"tolerance {tolerance:.4f}s at fps={aligned.fps}"
            )

    def test_run_timeline_raises_on_audio_drift(self, tmp_path, monkeypatch):
        """If the audio length drifts more than one frame from the
        aligned timeline (when ffprobe is available), ``run_timeline``
        must raise :class:`ValueError`. We patch the orchestrator's
        audio_probe call to FORCE a measurable drift so the test
        doesn't depend on ffprobe being installed.
        """
        synth_root = tmp_path / "synth_body_templates"
        _synth_template(
            synth_root, "alice", "idle",
            width=64, height=64, fps=25.0, total_frames=75,
            body_colour=(110, 110, 110), fmask_colour=(220, 0, 0),
        )
        _synth_template(
            synth_root, "alice", "explain_both",
            width=64, height=64, fps=25.0, total_frames=50,
            body_colour=(90, 90, 140), fmask_colour=(0, 220, 0),
        )

        # Synth an audio of EXACTLY 30 seconds (far outside the 8.0s
        # timeline + 1-frame tolerance at 25fps).
        audio_path = tmp_path / "speech_too_long.wav"
        _make_wav(audio_path, duration_s=30.0)

        timeline = Timeline.from_json(EXAMPLE_JSON)
        identity = AvatarIdentityHandle(
            identity_id=IdentityId("alice"),
            pack_path=tmp_path / "alice_pack",
            pack_digest="synth_digest",
            prepared_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        request = RenderRequest(
            job_id=RenderJobId("test_job_drift"),
            identity_id=IdentityId("alice"),
            identity_spec=IdentitySpec(source_image=tmp_path / "alice.png"),
            render_spec=RenderSpec(audio_path=audio_path),
        )

        orchestrator = RenderVideo(
            engine=_StubEngine(),
        )
        loader = _build_synth_loader(synth_root)
        # See comment in the happy-path test for why monkeypatch.setattr
        # (not raw __globals__ mutation) is the correct binding mechanism.
        monkeypatch.setattr(
            "src.application.render_video.use_case.load_body_template",
            loader,
        )

        # Force the audio probe to report a "drifted" duration so the
        # test runs deterministically regardless of ffprobe availability.
        monkeypatch.setattr(
            "src.application.render_video.use_case.audio_probe._probe_audio_duration",
            lambda _: 30.0,
        )

        with pytest.raises(ValueError) as exc:
            orchestrator.run_timeline(
                timeline, "alice", identity, request,
                alignment_dir=tmp_path / "aligned",
            )
        msg = str(exc.value)
        assert "drifts" in msg
        assert "aligned timeline" in msg


# ─────────────────────────────────────────────────────────────────────────────
# Minimal AvatarEngine stub — the integration test never invokes the engine
# because run_timeline's pathway stops at AlignedBodyTimeline construction.
# The orchestrator's ``run()`` (the engine-driven path) is not exercised here.
# ─────────────────────────────────────────────────────────────────────────────


class _StubEngine(AvatarEngine):
    """Stub satisfying the ABC contract the orchestrator types expect.

    Inherits from :class:`contracts.avatar_engine.AvatarEngine` so
    any future addition of ``engine.X()`` calls inside
    ``run_timeline`` fails LOUDLY with a TypeError on abstract
    method dispatch, rather than silently no-opping via duck
    typing.

    ``run_timeline`` does NOT call any engine method today — this
    stub exists purely to satisfy the ``RenderVideo(engine=...)``
    dataclass field type. All four abstract methods are implemented
    and raise :class:`NotImplementedError` so accidental future
    calls surface immediately rather than producing confusing
    mock artefacts.

    The dataclass setting a ``capture_dir`` is exposed via
    :attr:`settings` because ``render_video.use_case::_degraded_chunk_result``
    (the engine-driven ``run()`` path, NOT exercised here) writes
    fallback mp4s to ``engine.settings.capture_dir / job_id``. Even
    though we never reach that path under ``run_timeline``, the
    attribute is part of the orchestrator's downstream contract.
    """

    def __init__(self) -> None:
        self.engine_id = EngineId.LIVE_PORTRAIT
        from dataclasses import dataclass as _dataclass
        @_dataclass(slots=True)
        class _StubSettings:
            capture_dir: Path = Path("/tmp/integration_stub_captures")
        self.settings = _StubSettings()

    def load(self) -> None:
        raise NotImplementedError(
            "_StubEngine.load: run_timeline does not exercise the "
            "engine.load() pathway."
        )

    def unload(self) -> None:
        raise NotImplementedError(
            "_StubEngine.unload: run_timeline does not exercise the "
            "engine.unload() pathway."
        )

    def prepare_identity(self, source_image: Path) -> AvatarIdentityHandle:
        raise NotImplementedError(
            "_StubEngine.prepare_identity: run_timeline does not "
            "exercise the identity-prep pathway."
        )

    def render_chunk(
        self,
        chunk_req,
        identity: AvatarIdentityHandle,
    ):
        raise NotImplementedError(
            "_StubEngine.render_chunk: run_timeline does not "
            "exercise the engine.render_chunk() pathway."
        )
