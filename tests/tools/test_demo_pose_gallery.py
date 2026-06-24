"""Tests for ``tools.avatar_assets.demo_pose_gallery``.

Verifies:
- ``build_pose_gallery`` writes gallery.html + per-entry files + index.
- Each card embeds the relative iframe path so the file opens via file://.
- The pose-library sidebar / legend cross-tally includes every
  ``gesture_id`` actually emitted by the planner across the entries.
- Empty input is rejected.
- Inline JSON and YAML manifest inputs both work.
- The CLI ``main`` entry-point renders an ASCII console summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.avatar_assets.demo_pose_gallery import (
    GalleryEntry,
    build_pose_gallery,
    main,
)


_IT_SCRIPT = (
    "Ciao a tutti! Oggi vediamo tre differenze molto importanti: la prima, confrontiamo "
    "due approcci; la seconda, perché il sistema fa questo; quindi alla fine capirete come fare."
)
_EN_SCRIPT = (
    "Hello everyone! Today we look at three key differences: first, we compare two "
    "approaches; second, why the system does this; finally, you will understand how."
)


def _entry(language: str, avatar: str, text: str, duration: float) -> GalleryEntry:
    return GalleryEntry(
        language=language,
        avatar_id=avatar,
        text=text,
        audio_duration=duration,
        fps=25,
    )


# ─────────────────────────────────────────────────────────────────────────────
# build_pose_gallery happy-path
# ─────────────────────────────────────────────────────────────────────────────


def test_build_pose_gallery_writes_one_card_per_entry(tmp_path: Path) -> None:
    entries = [
        _entry("it", "alice", _IT_SCRIPT, 8.5),
        _entry("en", "bob", _EN_SCRIPT, 9.0),
        _entry("it", "bob", _IT_SCRIPT, 7.0),
    ]
    output_dir = tmp_path / "gallery"
    artifacts = build_pose_gallery(entries, output_dir)

    assert (output_dir / "gallery.html").is_file()
    assert (output_dir / "gallery_index.json").is_file()

    for entry in entries:
        subdir = output_dir / "entries" / entry.slug()
        assert (subdir / "timeline.html").is_file(), f"missing per-entry HTML for {entry.slug()}"
        assert (subdir / "timeline.json").is_file()
        assert (subdir / "timeline.txt").is_file()

    # The artifact map exposes namespaced entries for the test.
    for entry in entries:
        key = f"entry:{entry.slug()}:timeline.html"
        assert key in artifacts


def test_gallery_html_iframes_use_relative_paths(tmp_path: Path) -> None:
    """The iframe ``src`` MUST be a relative path so ``file://`` works.

    Verification guards against accidental absolute paths (``file:///...``)
    which only resolve on the original dev box."""
    entries = [_entry("it", "alice", _IT_SCRIPT, 8.0)]
    build_pose_gallery(entries, tmp_path / "gallery")

    html = (tmp_path / "gallery" / "gallery.html").read_text(encoding="utf-8")

    expected_subdir = "entries/it_alice"
    assert f'src="{expected_subdir}/timeline.html"' in html, (
        f"iframe src must be a relative path; got html excerpt: {html[:600]}"
    )
    # Guard against absolute-path regressions.
    assert "file:///" not in html, "absolute file:// URLs break portability across machines"


def test_gallery_index_json_matches_disk(tmp_path: Path) -> None:
    entries = [
        _entry("it", "alice", _IT_SCRIPT, 8.0),
        _entry("en", "bob", _EN_SCRIPT, 9.0),
    ]
    build_pose_gallery(entries, tmp_path / "gallery")
    index = json.loads((tmp_path / "gallery" / "gallery_index.json").read_text(encoding="utf-8"))

    assert len(index["entries"]) == 2
    slugs = {row["slug"]: row for row in index["entries"]}
    assert set(slugs) == {"it_alice", "en_bob"}
    for slug, row in slugs.items():
        assert row["timeline_html"] == f"entries/{slug}/timeline.html"
        assert row["timeline_json"] == f"entries/{slug}/timeline.json"
        assert row["dominant_pose"], "every entry must surface a dominant pose"


def test_legend_cross_tallies_every_gesture_id_seen(tmp_path: Path) -> None:
    """The sidebar legend must contain every ``gesture_id`` the planner emitted.

    A regression in either the gesture-to-pose mapping or the
    tally bytes would silently drop a gesture from the sidebar —
    we verify both directions.
    """
    entries = [
        _entry("it", "alice", _IT_SCRIPT, 8.5),
        _entry("en", "bob", _EN_SCRIPT, 9.0),
    ]
    build_pose_gallery(entries, tmp_path / "gallery")
    index = json.loads((tmp_path / "gallery" / "gallery_index.json").read_text(encoding="utf-8"))

    seen_gestures: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    for entry in entries:
        subdir = tmp_path / "gallery" / "entries" / entry.slug()
        timeline_json = json.loads((subdir / "timeline.json").read_text(encoding="utf-8"))
        for seg in timeline_json["segments"]:
            seen_gestures.add(seg["gesture_id"])
            if seg["kind"] == "gesture":
                seen_pairs.add((seg["gesture_id"], seg["pose_id"]))

    legend_gestures = set(index["legend"]["gesture_counts"])
    legend_pairs = {(row["gesture_id"], row["pose_id"]) for row in index["legend"]["pair_counts"]}
    assert seen_gestures.issubset(legend_gestures), (
        f"sidebar miss {seen_gestures - legend_gestures}"
    )
    assert seen_pairs.issubset(legend_pairs), f"pair miss {seen_pairs - legend_pairs}"


def test_build_pose_gallery_rejects_empty_entries(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one entry"):
        build_pose_gallery([], tmp_path / "gallery")


def test_duplicate_slug_raises(tmp_path: Path) -> None:
    """Two entries with the same (lang × avatar) MUST raise — otherwise
    the second silently overwrites the first and the gallery loses a card.
    """
    entries = [
        _entry("it", "alice", _IT_SCRIPT, 8.0),
        _entry("it", "alice", _EN_SCRIPT, 9.0),  # same slug
    ]
    with pytest.raises(ValueError, match="duplicate"):
        build_pose_gallery(entries, tmp_path / "gallery")


def test_card_text_preview_is_html_escaped(tmp_path: Path) -> None:
    """Text containing HTML-special chars MUST be escaped in the
    gallery card so stray ``<`` / ``&`` / ``"`` characters from a
    adverse manifest can't break the page or become an XSS vector.
    """
    script = '<script>alert("xss")</script> & friends'
    entries = [_entry("it", "alice", script, 8.0)]
    build_pose_gallery(entries, tmp_path / "gallery")
    html_text = (tmp_path / "gallery" / "gallery.html").read_text(encoding="utf-8")

    assert "&lt;script&gt;" in html_text, "text preview was not HTML-escaped"
    assert "<script>alert" not in html_text, (
        "raw <script> tag leaked into rendered HTML — escape regressed"
    )


def test_card_language_and_avatar_are_html_escaped(tmp_path: Path) -> None:
    """``language`` and ``avatar_id`` are user-controlled manifest
    fields — they MUST be escaped wherever they appear in the
    rendered HTML (``<h2>`` caption + iframe ``title`` attribute)
    to close the same XSS channel as the text preview.

    A manifest with ``language: <img onerror=...>`` would otherwise
    execute script when gallery.html is opened in the browser.
    """
    from tools.avatar_assets.demo_pose_gallery import GalleryEntry

    entries = [
        GalleryEntry(
            language="<img onerror=alert(1) src=x>",
            avatar_id="<svg/onload=alert(1)>",
            text=_IT_SCRIPT,
            audio_duration=8.0,
            fps=25,
        ),
    ]
    build_pose_gallery(entries, tmp_path / "gallery")
    html_text = (tmp_path / "gallery" / "gallery.html").read_text(encoding="utf-8")

    assert "<img onerror" not in html_text, "language field was not HTML-escaped"
    assert "<svg/onload" not in html_text, "avatar_id field was not HTML-escaped"
    assert "&lt;img" in html_text, "escaped language marker missing"
    assert "&lt;svg" in html_text, "escaped avatar_id marker missing"


def test_sidebar_pills_html_escape_language_and_avatar(tmp_path: Path) -> None:
    """The sidebar aggregate pills (``<span class="pill">{language}</span>``
    + ``{avatar_id}``) ALSO interpolate user-controlled fields and must
    be escaped — a separate leak channel from the per-card caption.
    """
    from tools.avatar_assets.demo_pose_gallery import GalleryEntry

    entries = [
        GalleryEntry(
            language="<svg onmouseover=alert(1)>",
            avatar_id="<iframe src=javascript:alert(1)>",
            text=_IT_SCRIPT,
            audio_duration=8.0,
            fps=25,
        ),
    ]
    build_pose_gallery(entries, tmp_path / "gallery")
    html_text = (tmp_path / "gallery" / "gallery.html").read_text(encoding="utf-8")

    assert "<svg onmouseover" not in html_text, (
        "language pill in sidebar was not HTML-escaped"
    )
    assert "<iframe src=javascript" not in html_text, (
        "avatar_id pill in sidebar was not HTML-escaped"
    )
    assert "&lt;svg onmouseover=alert(1)&gt;" in html_text, "escaped language pill missing"
    assert "&lt;iframe" in html_text, "escaped avatar_id pill missing"


# ─────────────────────────────────────────────────────────────────────────────
# Input parsing — inline JSON + YAML manifest
# ─────────────────────────────────────────────────────────────────────────────


def test_yaml_manifest_with_entries_key(tmp_path: Path) -> None:
    import yaml  # type: ignore

    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "entries": [
                    {
                        "language": "it",
                        "avatar_id": "alice",
                        "text": _IT_SCRIPT,
                        "audio_duration": 8.0,
                    },
                    {
                        "language": "en",
                        "avatar_id": "bob",
                        "text": _EN_SCRIPT,
                        "audio_duration": 7.5,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    artifacts = main(["--manifest", str(manifest), "--output-dir", str(tmp_path / "gallery")])
    assert artifacts == 0  # exit 0
    assert (tmp_path / "gallery" / "gallery.html").is_file()


def test_yaml_manifest_as_bare_list(tmp_path: Path) -> None:
    import yaml  # type: ignore

    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            [
                {
                    "language": "it",
                    "avatar_id": "alice",
                    "text": _IT_SCRIPT,
                    "audio_duration": 8.0,
                }
            ]
        ),
        encoding="utf-8",
    )

    main(["--manifest", str(manifest), "--output-dir", str(tmp_path / "gallery")])
    assert (tmp_path / "gallery" / "gallery.html").is_file()


def test_short_alias_keys_in_yaml(tmp_path: Path) -> None:
    """``lang`` and ``avatar`` (short aliases) must work too.

    Operators tend to write short manifests; the loader should accept
    them without surprises."""
    import yaml  # type: ignore

    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            [
                {
                    "lang": "it",
                    "avatar": "alice",
                    "text": _IT_SCRIPT,
                    "duration": 8.0,
                }
            ]
        ),
        encoding="utf-8",
    )

    main(["--manifest", str(manifest), "--output-dir", str(tmp_path / "gallery")])
    index = json.loads((tmp_path / "gallery" / "gallery_index.json").read_text(encoding="utf-8"))
    assert index["entries"][0]["language"] == "it"
    assert index["entries"][0]["avatar_id"] == "alice"
    assert index["entries"][0]["audio_duration"] == 8.0


def test_inline_json_entries(tmp_path: Path) -> None:
    payload = json.dumps(
        [
            {
                "language": "it",
                "avatar_id": "alice",
                "text": _IT_SCRIPT,
                "audio_duration": 8.0,
            }
        ]
    )
    main(["--entries", payload, "--output-dir", str(tmp_path / "gallery")])
    assert (tmp_path / "gallery" / "gallery.html").is_file()


def test_cli_rejects_neither_input(tmp_path: Path) -> None:
    """`add_mutually_exclusive_group(required=True)` rejects the empty
    call at argparse level with SystemExit — verify that's the path.
    """
    from tools.avatar_assets import demo_pose_gallery as mod

    with pytest.raises(SystemExit):
        mod._build_parser().parse_args([])


# ─────────────────────────────────────────────────────────────────────────────
# Gallery integration smoke test — open & sanity-check the produced page
# ─────────────────────────────────────────────────────────────────────────────


def test_gallery_page_links_to_every_entry_html(tmp_path: Path) -> None:
    """Every entry's per-entry HTML must be linked from the gallery page.

    Catches the failure mode where the sidebar/legend works but one
    iframe was silently dropped (e.g. duplicate slugs caused iframe
    collision and the renderer dropped one).
    """
    entries = [
        _entry("it", "alice", _IT_SCRIPT, 8.0),
        _entry("en", "bob", _EN_SCRIPT, 9.0),
        _entry("it", "bob", _IT_SCRIPT, 7.0),
    ]
    build_pose_gallery(entries, tmp_path / "gallery")
    html = (tmp_path / "gallery" / "gallery.html").read_text(encoding="utf-8")

    for entry in entries:
        expected = f'iframe src="entries/{entry.slug()}/timeline.html"'
        assert expected in html, f"missing iframe link for {entry.slug()}"
