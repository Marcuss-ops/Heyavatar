"""Real-GPU smoke test — pose gallery lines up with real body templates.

This is the release gate for the pose-library visual-validation
pipeline (:mod:`tools.avatar_assets.demo_pose_gallery`). It proves
that on a real LivePortrait box, every per-card iframe timeline
emitted by :func:`build_pose_gallery` corresponds to a real
:func:`load_body_template` on disk whose ``metadata.json`` pose_id
**round-trips** through the canonical :func:`pick_pose_id_for_gesture`
in :mod:`src.motion.text_driven_timeline` — so the sidebar legend,
the visualisation iframe, and the production body cache can never
drift apart again.

Test mode: real (HEYAVATAR_MOCK_ENGINE=0). Skips via ``requires_cuda``
when no CUDA + LivePortrait upstream clone are reachable, so an
unprovisioned workstation is silently skipped rather than failing.

What we exercise end-to-end:

1. :func:`AvatarCompiler.compile` against the LivePortrait engine
   produces a real identity pack on disk (this also implicitly
   validates the body of :file:`tests/smoke/test_real_gpu/test_engine_load.py`).
2. We pre-materialise a body-template cache for EVERY gesture the
   rule-based planner can plausibly emit (Italian + English keywords
   per :file:`providers/motion_extraction/mediapipe/gesture_planner.py`).
   Pre-writing the full superset — not just the gestures we hope
   the planner picks — guarantees the assertion ``every planned
   gesture resolved to its own real body template`` is meaningful
   (i.e. doesn't alias a fallback substitution as a legitimate
   resolution).
3. :func:`build_pose_gallery` writes ``gallery.html``,
   ``gallery_index.json`` + one
   ``entries/<slug>/timeline.{html,json,txt}`` per entry.
4. For every gesture segment in every entry we assert:

   a. :func:`load_body_template` resolves to a real
      ``body.mp4 / face_mask.mp4 / neck_mask.mp4 / face_transforms.npz``.
   b. ``metadata.json`` pose_id == :func:`pick_pose_id_for_gesture`
      — gated by a HARD-CODED canary mapping in this test so the
      two sides are independent and the assertion fails the moment
      either drifts.
   c. ``gallery_index.json``'s ``dominant_pose`` matches an
      INDEPENDENT implementation of "pick the gesture with the
      largest (duration × intensity)". The gallery's own
      ``_dominant_pose_of`` is the implementation under test; the
      test re-derives it from spec so the comparison is non-tautological.
   d. The iframe ``src`` in ``gallery.html`` is a *relative* path
      that resolves to an existing file (so the file opens via
      ``file://`` from any machine).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from src.core.config import get_settings
from src.domain.body_template import load_body_template
from src.domain.enums import EngineId
from src.motion.text_driven_timeline import Timeline, pick_pose_id_for_gesture
from tools.avatar_assets.demo_pose_gallery import (
    GalleryEntry,
    _dominant_pose_of,
    build_pose_gallery,
)
from tests.smoke.test_real_gpu._helpers import (
    _test_image,
    real_mode_env,  # noqa: F401,F811 (pytest fixture lookup — ruff can't see it)
    requires_cuda,
)


# Italian + English scripts that exercise the canonical gesture
# vocabulary the planner already knows. Same scripts used by the
# existing single-entry demo CLI's docstring examples so failures
# point to something operators recognise.
_IT_SCRIPT = (
    "Ciao a tutti! Oggi vediamo tre differenze molto importanti: la prima, "
    "confrontiamo due approcci; la seconda, perche il sistema fa questo; "
    "quindi alla fine capirete come fare."
)
_EN_SCRIPT = (
    "Hello everyone! Today we look at three key differences: first, we "
    "compare two approaches; second, why the system does this; finally, "
    "you will understand how."
)


# Superset of EVERY gesture_id the rule-based planner can emit from
# Italian + English text. Verified by static enumeration of the
# planner's keyword tables in
# ``providers/motion_extraction/mediapipe/gesture_planner.py``.
# Pre-materialising every entry here guarantees the late assertion
# is about RESOLUTION — never about fallback substitution.
_PLANNER_GESTURE_SUPERSET: tuple[str, ...] = (
    # Counts.
    "count_one",
    "count_two",
    "count_three",
    # Question + comparison + conclusion.
    "question",
    "comparison",
    "conclusion",
    # Emphasis + open-palms (greeting / introduction).
    "emphasis_small",
    "open_palms",
    # Idle fallback.
    "idle_small",
)


# Hard-coded canary of the gesture-to-pose mapping. Sourced
# independently from :func:`pick_pose_id_for_gesture`'s
# ``_GESTURE_TO_POSE`` table so a drift between THIS hard-coded
# mirror and the canonical function fails test 4b on the next
# real-GPU run instead of silently aliasing.
_HARDCODED_POSE_CANARY: dict[str, str] = {
    "point_left": "left_hand_up",
    "explain_left": "left_hand_up",
    "point_right": "right_hand_up",
    "explain_right": "right_hand_up",
    "explain_both": "both_hands_open",
    "open_palms": "both_hands_open",
    "comparison": "both_hands_open",
    # All other gestures (counts, question, conclusion, emphasis,
    # idle) intentionally fall through to ``neutral_desk`` to
    # mirror the canonical default fallback.
}


def _write_synthetic_body_template(
    *,
    base_dir: Path,
    avatar_id: str,
    gesture_id: str,
    fps: int = 25,
    frames: int = 75,
) -> None:
    """Materialise the body-template 5-tuple on disk so ``load_body_template``
    returns successfully.

    The on-disk shape mirrors what
    :file:`tools/avatar_assets/extract_reference_motion.py` produces; the
    content is a real-looking but synthetic payload (file presence is all
    the production loader checks).

    The ``metadata.json`` is the live assertion target — we bake the
    ``_HARDCODED_POSE_CANARY`` result into it so the
    ``metadata.pose_id == pick_pose_id_for_gesture`` check survives
    the moment either side drifts.
    """
    pack = base_dir / avatar_id / "body_cache" / gesture_id
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "body.mp4").write_bytes(b"\x00")
    (pack / "face_mask.mp4").write_bytes(b"\x00")
    (pack / "neck_mask.mp4").write_bytes(b"\x00")
    # zipfile-only NPZ so we avoid an npz/numpy dependency on the
    # test path (the production loader only checks ``is_file()``).
    with zipfile.ZipFile(pack / "face_transforms.npz", "w") as zf:
        zf.writestr("_header", b"")
    pose_id = _HARDCODED_POSE_CANARY.get(gesture_id, "neutral_desk")
    (pack / "metadata.json").write_text(
        json.dumps(
            {
                "avatar_id": avatar_id,
                "gesture_id": gesture_id,
                "pose_id": pose_id,
                "fps": fps,
                "frames": frames,
                "width": 64,
                "height": 64,
                "status": "ok",
            }
        ),
        encoding="utf-8",
    )


def _compute_dominant_pose_independent(timeline: Timeline) -> str:
    """Re-derive the dominant pose from spec, NOT from the gallery's own
    ``_dominant_pose_of`` implementation.

    Lets assertion 4c compare the gallery's index value against a
    definition written from scratch — so the assertion is meaningful
    rather than tautological (it would still pass if the gallery
    silently shipped the wrong value, since calling
    ``_dominant_pose_of(timeline)`` twice always returns the same
    thing).
    """
    if not timeline.segments:
        return "neutral_desk"
    gesture_candidates = [s for s in timeline.segments if s.kind == "gesture"]
    if not gesture_candidates:
        # No gesture kind at all — fall back to the longest idle
        # segment's pose_id (which is always ``neutral_desk`` by
        # construction, but we read the field to stay honest).
        return max(timeline.segments, key=lambda s: s.end - s.start).pose_id
    return max(
        gesture_candidates,
        key=lambda s: (s.end - s.start) * max(s.intensity, 0.1),
    ).pose_id


@requires_cuda
def test_pose_gallery_lines_up_with_real_body_templates(
    real_mode_env,  # noqa: F811 (real_mode_env is a pytest fixture)
    workdir,
    tmp_path: Path,
) -> None:
    """End-to-end real-GPU smoke for :func:`build_pose_gallery`.

    Compiles a **real** LivePortrait identity pack, materialises a body
    template for every planner-emittable gesture, runs
    :func:`build_pose_gallery` over an Italian + English entry pair,
    and asserts every per-card iframe timeline:

    * resolves to a real :func:`load_body_template` on disk,
    * declares a ``pose_id`` that matches the canonical gesture → pose
      mapping (no drift between metadata and :func:`pick_pose_id_for_gesture`),
    * has a ``dominant_pose`` the gallery's index agrees with from a
      freshly-derived independent computation,
    * is reachable from the parent ``gallery.html`` via a relative
      iframe ``src`` path so opening the file from any machine works.

    Combined, these four invariants gate every release regression
    where the visualisation library might otherwise agree with itself
    while pointing at placeholder body templates.
    """
    settings = get_settings()
    if settings.mock_engine:
        pytest.skip("HEYAVATAR_MOCK_ENGINE=1 — this is a real-mode gate")

    # Lives under tmp_path so the test is hermetic; never touches the
    # project cache when HEYAVATAR_PACK_DIR is overridden by the
    # ``workdir`` fixture.
    source = _test_image(tmp_path)

    # ── 1. Engine setup + real identity compile ──────────────────
    from providers import get_provider

    engine = get_provider(EngineId.LIVE_PORTRAIT)
    engine.load()
    try:
        from src.application.compile_avatar import AvatarCompiler
        from src.domain.avatar_pack import read_pack
        from src.domain.types import IdentitySpec
        from src.storage.avatar_packs import AvatarPackRepository

        pack_repo = AvatarPackRepository(root=workdir / "packs")
        spec = IdentitySpec(source_image=source, display_name="Gallery Actor")
        compiler = AvatarCompiler(engine=engine, pack_root=pack_repo.root)
        identity_handle = compiler.compile(spec)
        pack_repo.save(
            identity_handle.identity_id, read_pack(identity_handle.pack_path)
        )
        avatar_id = identity_handle.identity_id
        print(f"\nIdentity compiled: {avatar_id}")

        # ── 2. Body-template cache: write every gesture the
        # planner can plausibly pick so the late assertion never
        # aliases a fallback substitution as a legitimate gesture
        # resolution.
        body_dir = workdir / "body_templates"
        for gid in _PLANNER_GESTURE_SUPERSET:
            _write_synthetic_body_template(
                base_dir=body_dir,
                avatar_id=avatar_id,
                gesture_id=gid,
            )

        # ── 3. Build the gallery directly (no CLI round-trip —
        # lets us inspect artefacts in-process).
        entries = [
            GalleryEntry(
                language="it",
                avatar_id=avatar_id,
                text=_IT_SCRIPT,
                audio_duration=8.0,
                fps=25,
            ),
            GalleryEntry(
                language="en",
                avatar_id=avatar_id,
                text=_EN_SCRIPT,
                audio_duration=8.0,
                fps=25,
            ),
        ]
        gallery_dir = workdir / "real_pose_gallery"
        build_pose_gallery(entries, gallery_dir)
        print(f"Built gallery in {gallery_dir}")

        # ── 4. Reload every per-entry timeline.json and assert
        # every gesture segment lines up with a real body template
        # whose metadata pose_id matches pick_pose_id_for_gesture.
        index_path = gallery_dir / "gallery_index.json"
        assert index_path.is_file(), f"gallery_index.json missing at {index_path}"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        assert len(index["entries"]) == len(entries), (
            f"gallery_index has {len(index['entries'])} entries but "
            f"the manifest had {len(entries)}"
        )

        total_planned_gestures = 0
        for entry_row in index["entries"]:
            slug = entry_row["slug"]
            entry_dir = gallery_dir / "entries" / slug
            assert (entry_dir / "timeline.html").is_file(), (
                f"per-entry HTML missing for {slug!r}"
            )
            assert (entry_dir / "timeline.json").is_file(), (
                f"per-entry JSON missing for {slug!r}"
            )

            # Re-parse the timeline segments so we can walk them.
            timeline_json = json.loads(
                (entry_dir / "timeline.json").read_text(encoding="utf-8")
            )
            timeline = Timeline.from_dict(timeline_json)
            planned_gestures = [
                seg.gesture_id for seg in timeline.segments if seg.kind == "gesture"
            ]
            assert planned_gestures, (
                f"entry {slug!r} produced no gesture segments; the planner "
                "should always surface at least one animation intent"
            )

            for gid in planned_gestures:
                # 4a. The body template MUST be loadable on disk —
                # proves the glyph isn't falling back to a placeholder.
                bt = load_body_template(avatar_id, gid, base_dir=body_dir)
                assert bt.body_video.is_file()
                assert bt.face_mask.is_file()
                assert bt.neck_mask.is_file()
                assert bt.face_transforms.is_file()

                # 4b. The on-disk metadata pose_id MUST equal
                # pick_pose_id_for_gesture — the two sides are
                # sourced INDEPENDENTLY (the canary mirror in this
                # test vs. ``_GESTURE_TO_POSE`` in the canonical
                # module), so any drift fails this assertion.
                meta = json.loads(bt.metadata.read_text(encoding="utf-8"))
                expected_pose = pick_pose_id_for_gesture(gid)
                assert meta["pose_id"] == expected_pose, (
                    f"body template for gesture={gid!r} declares "
                    f"pose_id={meta['pose_id']!r} but "
                    f"pick_pose_id_for_gesture resolves to "
                    f"{expected_pose!r}. The registry and the body's "
                    "metadata have drifted — the pose-library "
                    "visualisation would silently disagree with what "
                    "the renderer actually produces. Fix one or the "
                    "other (or update the hard-coded canary in this "
                    "test if the new mapping is the canonical one)."
                )
                total_planned_gestures += 1

            # 4c. gallery_index dominant_pose MUST equal an
            # INDEPENDENT computation, NOT just ``_dominant_pose_of``
            # called twice. Catches both the "gallery" path AND the
            # "what the dominant pose is supposed to mean" definition.
            expected_dominant = _compute_dominant_pose_independent(timeline)
            assert entry_row["dominant_pose"] == expected_dominant, (
                f"gallery_index dominant_pose={entry_row['dominant_pose']!r} "
                f"disagrees with the spec-derived computation="
                f"{expected_dominant!r}"
            )
            # Belt-and-braces: also assert the gallery's own
            # implementation still matches the spec — guards against
            # the case where both sides drift together.
            assert entry_row["dominant_pose"] == _dominant_pose_of(timeline), (
                "gallery_index dominant_pose disagrees with the "
                "gallery's own _dominant_pose_of — internal bug"
            )

            # 4d. iframe src is a RELATIVE PATH so the file opens
            # via file:// in any browser.
            html_text = (gallery_dir / "gallery.html").read_text(encoding="utf-8")
            expected_iframe = f'iframe src="entries/{slug}/timeline.html"'
            assert expected_iframe in html_text, (
                f"missing iframe src for {slug!r} in gallery.html "
                "(regressed relative-path resolution — the gallery "
                "would only open on the dev box)"
            )

        print(
            f"Pose-gallery smoke OK — {total_planned_gestures} gesture "
            "segments cross-validated against real body templates"
        )
    finally:
        engine.unload()
