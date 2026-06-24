"""Demo CLI — pose library gallery built from a list of (lang × avatar) entries.

This script is the third demo artifact the slim plan §6 timeline
section calls out: it takes a small manifest of (language,
avatar_id, text, audio_duration) tuples, runs each through the
existing demo-gesture-timeline pipeline, and emits a parent
``gallery.html`` that embeds every per-avatar result as a self-
contained iframe card. The result is a single HTML file an operator
can open in the browser to validate the **pose library** at a
glance:

- Does the planner pick the gestures we expected for this text?
- Do the same intent keywords produce equivalent gesture_ids in
  two different languages?
- Which ``pose_id`` is each ``gesture_id`` mapped to across the
  whole dataset?

Each card embeds the existing ``timeline.html`` self-contained
artifact (CSS @keyframes animation of an SVG stick figure) — no
SMIL, no JS, no rewriting of the per-entry HTML. The parent
gallery page only adds a captions strip + a pose-library sidebar
that cross-checks every gesture_id / pose_id combination the
planner emitted for the manifest, so a regression in the
gesture-to-pose mapping shows up immediately next to the
animation.

Inputs
------
- ``--entries``: inline JSON list of entries.
- ``--manifest``: path to a YAML file (preferred when more than
  2-3 entries; matches the project's YAML-everywhere convention
  for ``registry/*.yaml``).

Output
------
- ``gallery.html`` (responsive CSS grid of iframes + sidebar)
- ``entries/<lang>_<avatar>/timeline.html`` (one per entry, each
  fully self-contained)
- ``gallery_index.json`` (cross-entry gesture/pose summary, handy
  for CI assertions)

Run::

    python tools/avatar_assets/demo_pose_gallery.py \\
      --manifest docs/gallery_example.yaml \\
      --output-dir captures/pose_gallery

The script is import-safe on a stock Python install — only
``pyyaml`` is added to the dependency footprint of the existing
single-entry CLI.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Bootstrap sys.path so the script works in two invocation styles:
#   1. `python tools/avatar_assets/demo_pose_gallery.py …` from project root
#   2. `python -m tools.avatar_assets.demo_pose_gallery …`
# Without this, sys.path[0] is the script's own directory (tools/avatar_assets/)
# and the `from tools…` import line below fails. Operators shouldn't need to
# `pip install -e .` just to run a demo CLI.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.motion.text_driven_timeline import Timeline, TimelineSegment, text_to_timeline

from tools.avatar_assets.demo_gesture_timeline import (  # noqa: E402  (after sys.path bootstrap)
    _format_ascii,
    _format_html,
    _resolve_segment_summary,
)


# ─────────────────────────────────────────────────────────────────────────────
# Entry schema
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class GalleryEntry:
    """One row of the gallery manifest.

    Each entry maps 1:1 to a per-entry ``timeline.html`` written
    under ``<output-dir>/entries/<language>_<avatar_id>/``.
    """

    language: str
    avatar_id: str
    text: str
    audio_duration: float
    fps: int = 25

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "GalleryEntry":
        # Accept both ``language`` and ``lang`` (shorter alias for
        # terse YAMLs); ``avatar`` and ``avatar_id`` likewise.
        language = str(payload.get("language") or payload.get("lang") or "it")
        avatar_id = str(payload.get("avatar_id") or payload.get("avatar") or "default")
        text = str(payload["text"])
        duration = float(payload.get("audio_duration") or payload.get("duration") or 8.0)
        fps = int(payload.get("fps", 25))
        return cls(language=language, avatar_id=avatar_id, text=text, audio_duration=duration, fps=fps)

    def slug(self) -> str:
        """Filesystem-friendly shorthand used for entry subdirs + iframes."""
        return _slugify(f"{self.language}_{self.avatar_id}")


def _slugify(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_").lower()


# ─────────────────────────────────────────────────────────────────────────────
# Entry parsing — JSON inline OR YAML manifest
# ─────────────────────────────────────────────────────────────────────────────


def _load_yaml_entries(path: Path) -> list[GalleryEntry]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - covered on real installs
        raise RuntimeError(
            "pyyaml is required for --manifest; install with `pip install pyyaml`"
        ) from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    rows: Iterable[Mapping[str, Any]]
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and "entries" in payload:
        rows = payload["entries"]
    else:
        raise ValueError(
            f"{path} must be a list of entries or {{'entries': [...]}}"
        )
    return [GalleryEntry.from_mapping(row) for row in rows]


def _load_json_entries(value: str) -> list[GalleryEntry]:
    payload = json.loads(value)
    if not isinstance(payload, list):
        raise ValueError("--entries JSON must be a list of entry objects")
    return [GalleryEntry.from_mapping(row) for row in payload]


# ─────────────────────────────────────────────────────────────────────────────
# Per-entry artifact writer — delegates to the existing demo CLI primitives
# ─────────────────────────────────────────────────────────────────────────────


def _write_entry_artifact(
    entry: GalleryEntry, subdir: Path
) -> tuple[Path, Path, Path, Timeline]:
    """Write the existing 4-file artifact set for one entry.

    Returns (json_path, txt_path, html_path, timeline) so the
    gallery-index builder can reuse the parsed Timeline without a
    second round-trip through ``text_to_timeline``.
    """
    subdir.mkdir(parents=True, exist_ok=True)
    timeline = text_to_timeline(
        text=entry.text,
        audio_duration=entry.audio_duration,
        avatar_id=entry.avatar_id,
        language=entry.language,
        fps=entry.fps,
    )

    json_path = subdir / "timeline.json"
    txt_path = subdir / "timeline.txt"
    html_path = subdir / "timeline.html"

    payload = timeline.to_dict()
    payload["entry"] = {
        "language": entry.language,
        "avatar_id": entry.avatar_id,
        "audio_duration": entry.audio_duration,
        "fps": entry.fps,
        "text_preview": entry.text[:160],
    }
    payload["summary"] = _resolve_segment_summary(timeline.segments)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    txt_path.write_text(_format_ascii(timeline), encoding="utf-8")

    html = _format_html(timeline)
    html = html.replace("$(duration)", f"{timeline.duration:.2f}")
    first_gesture = timeline.segments[0].gesture_id if timeline.segments else "-"
    html = html.replace("$(first_gesture)", first_gesture)
    html_path.write_text(html, encoding="utf-8")
    return json_path, txt_path, html_path, timeline


# ─────────────────────────────────────────────────────────────────────────────
# Cross-entry pose library summary
# ─────────────────────────────────────────────────────────────────────────────


def _dominant_pose_of(timeline: Timeline) -> str:
    """Return the pose_id of the segment with the largest (duration * intensity).

    Used by both :func:`_entry_card_html` and :func:`_build_gallery_index`
    so the card caption and the index JSON agree on which pose the
    timeline actually peaks at. Idle segments count as fallback
    only — a real gesture segment always wins if it's longer or
    more intense.
    """
    if not timeline.segments:
        return "neutral_desk"

    def _key(seg: TimelineSegment) -> tuple[float, int]:
        # Gesture-kind segments always ranked above idle-kind so a
        # long idle won't beat a short strong gesture.
        return ((seg.end - seg.start) * max(seg.intensity, 0.1), 1 if seg.kind == "gesture" else 0)

    return max(timeline.segments, key=_key).pose_id


def _build_pose_legend(entries: Sequence[tuple[GalleryEntry, Timeline]]) -> dict[str, Any]:
    """Cross-tally gesture_id → pose_id and frequency across all entries.

    The legend serves two purposes:
    1. The sidebar in ``gallery.html`` shows it visually so a
       reviewer can spot ``gesture_id`` → ``pose_id`` regressions at
       a glance.
    2. ``gallery_index.json`` ships the same data so CI can assert
       that ``every planned gesture_id resolves to a known pose_id``.
    """
    gesture_counts: dict[str, int] = {}
    pose_counts: dict[str, int] = {}
    pair_counts: dict[tuple[str, str], int] = {}
    for _entry, timeline in entries:
        for seg in timeline.segments:
            gesture_counts[seg.gesture_id] = gesture_counts.get(seg.gesture_id, 0) + 1
            if seg.kind == "gesture":
                pose_counts[seg.pose_id] = pose_counts.get(seg.pose_id, 0) + 1
                pair_counts[(seg.gesture_id, seg.pose_id)] = (
                    pair_counts.get((seg.gesture_id, seg.pose_id), 0) + 1
                )
    return {
        "gesture_counts": dict(sorted(gesture_counts.items())),
        "pose_counts": dict(sorted(pose_counts.items())),
        "pair_counts": [
            {"gesture_id": g, "pose_id": p, "count": c}
            for (g, p), c in sorted(pair_counts.items())
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# gallery.html — parent page with an iframe grid + a pose-library sidebar
# ─────────────────────────────────────────────────────────────────────────────


_GALLERY_CSS = """
:root {
  --bg: #f7f7f4;
  --fg: #222;
  --muted: #555;
  --accent: #2473ff;
  --card-bg: #ffffff;
  --card-border: #d9d9d9;
  --legend-bg: #f0f4fa;
}
body {
  font: 15px/1.4 system-ui, sans-serif;
  background: var(--bg); color: var(--fg); margin: 24px;
}
header { margin-bottom: 18px; }
header h1 { margin: 0 0 4px 0; }
header p { margin: 0; color: var(--muted); font-size: 13px; }
.gallery-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
  gap: 18px;
}
.card {
  background: var(--card-bg); border: 1px solid var(--card-border);
  border-radius: 10px; overflow: hidden;
  display: flex; flex-direction: column;
}
.card header {
  padding: 10px 14px; border-bottom: 1px solid var(--card-border);
  background: #fafaf7;
}
.card header h2 { margin: 0 0 4px 0; font-size: 15px; }
.card header .meta { color: var(--muted); font-size: 12px; }
.card .frame-wrap { background: #ededed; }
.card iframe { width: 100%; height: 520px; border: 0; display: block; }
.layout { display: grid; grid-template-columns: minmax(0, 3fr) minmax(220px, 1fr); gap: 18px; }
@media (max-width: 900px) { .layout { grid-template-columns: 1fr; } }
.sidebar {
  background: var(--legend-bg); border: 1px solid var(--card-border);
  border-radius: 10px; padding: 14px;
  position: sticky; top: 12px; align-self: start; max-height: calc(100vh - 24px); overflow: auto;
}
.sidebar h3 { margin: 0 0 6px 0; font-size: 14px; }
.sidebar dl { margin: 0; }
.sidebar dt { font-weight: 600; }
.sidebar dd { margin: 0 0 4px 0; color: var(--muted); font-size: 13px; }
.pill {
  display: inline-block; padding: 2px 8px; border-radius: 999px;
  background: #fff; border: 1px solid #ccc; font-size: 12px; margin: 0 4px 4px 0;
}
.pill-strong { background: var(--accent); color: #fff; border-color: var(--accent); }
table.pair-table { width: 100%; border-collapse: collapse; font-size: 12px; }
table.pair-table th, table.pair-table td { padding: 4px 6px; border-bottom: 1px solid #d6dde8; text-align: left; }
table.pair-table th { background: #dde6f2; }
.tag {
  display: inline-block; padding: 1px 6px; border-radius: 4px; background: #dde6f2; font-size: 11px;
}
"""


def _entry_card_html(entry: GalleryEntry, subdir_name: str, timeline: Timeline) -> str:
    dominant_label = _dominant_pose_of(timeline)
    language = html.escape(entry.language, quote=True)
    avatar_id = html.escape(entry.avatar_id, quote=True)
    text_preview = html.escape(entry.text.strip(), quote=True)
    if len(text_preview) > 96:
        text_preview = text_preview[:93] + "…"
    return f"""
  <article class="card" id="card-{entry.slug()}">
    <header>
      <h2>{language} · {avatar_id}</h2>
      <div class="meta">
        {timeline.duration:.2f}s · {len(timeline.segments)} segments ·
        fps {entry.fps} ·
        <span class="tag">dominant pose: <b>{dominant_label}</b></span>
      </div>
    </header>
    <div class="frame-wrap">
      <iframe src="entries/{subdir_name}/timeline.html" title="{language} / {avatar_id} timeline"
              loading="lazy"></iframe>
    </div>
    <footer style="padding: 8px 14px; background: #fafaf7; font-size: 12px; color: var(--muted);">
      “{text_preview}”
    </footer>
  </article>
"""


def _sidebar_html(legend: Mapping[str, Any], entries: Sequence[tuple[GalleryEntry, Timeline]]) -> str:
    gestures_html = " ".join(
        f'<span class="pill{"" if count == 1 else " pill-strong"}">{gid} × {count}</span>'
        for gid, count in legend["gesture_counts"].items()
    )
    poses_html = "\n".join(
        f"<dt>{pose}</dt><dd>{count}×</dd>"
        for pose, count in legend["pose_counts"].items()
    ) or "<dt>(none)</dt><dd>0</dd>"
    pair_rows = "\n".join(
        f"<tr><td>{row['gesture_id']}</td><td>→</td><td>{row['pose_id']}</td><td>{row['count']}</td></tr>"
        for row in legend["pair_counts"]
    ) or "<tr><td colspan='4'>(no gesture segments)</td></tr>"
    entry_count = len(entries)
    languages = sorted({e.language for e, _ in entries})
    avatars = sorted({e.avatar_id for e, _ in entries})
    header = f"<p>{entry_count} entries · {len(languages)} languages · {len(avatars)} avatars</p>"
    languages_html = " ".join(f'<span class="pill">{html.escape(l, quote=True)}</span>' for l in languages)
    avatars_html = " ".join(f'<span class="pill">{html.escape(a, quote=True)}</span>' for a in avatars)
    return f"""
  <aside class="sidebar">
    <h3>Pose library legend</h3>
    {header}
    <p><b>Gestures seen</b><br/>{gestures_html}</p>
    <p><b>Poses seen</b><br/><dl>{poses_html}</dl></p>
    <p><b>gesture → pose mapping</b></p>
    <table class="pair-table"><thead><tr><th>gesture</th><th></th><th>pose</th><th>count</th></tr></thead>
      <tbody>{pair_rows}</tbody></table>
    <p><b>Languages</b> {languages_html}</p>
    <p><b>Avatars</b> {avatars_html}</p>
  </aside>
"""


def _build_gallery_html(
    entries_with_timelines: Sequence[tuple[GalleryEntry, Timeline]],
    output_dir: Path,
    legend: Mapping[str, Any],
) -> Path:
    """Compose the parent gallery.html in ``output_dir``.

    Each entry's iframe references ``entries/<slug>/timeline.html``,
    a RELATIVE path so opening the file directly via file:// works
    on every browser without needing a local webserver.
    """
    cards = "\n".join(
        _entry_card_html(entry, entry.slug(), timeline)
        for entry, timeline in entries_with_timelines
    )
    sidebar = _sidebar_html(legend, entries_with_timelines)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>heyavatar pose library gallery</title>
  <style>{_GALLERY_CSS}</style>
</head>
<body>
  <header>
    <h1>heyavatar pose library gallery</h1>
    <p>Each card embeds a per-(language × avatar) timeline animation. Use the sidebar to cross-check
       every <code>gesture_id</code> against its resolved <code>pose_id</code> across the whole dataset.</p>
  </header>
  <div class="layout">
    <main>
      <section class="gallery-grid">
        {cards}
      </section>
    </main>
    {sidebar}
  </div>
</body>
</html>"""
    gallery_path = output_dir / "gallery.html"
    output_dir.mkdir(parents=True, exist_ok=True)
    gallery_path.write_text(html, encoding="utf-8")
    return gallery_path


# ─────────────────────────────────────────────────────────────────────────────
# Index file (CI-friendly assertion target)
# ─────────────────────────────────────────────────────────────────────────────


def _build_gallery_index(
    entries_with_timelines: Sequence[tuple[GalleryEntry, Timeline]],
    legend: Mapping[str, Any],
    gallery_path: Path,
) -> Path:
    payload = {
        "gallery_html": str(gallery_path),
        "entries": [
            {
                "language": e.language,
                "avatar_id": e.avatar_id,
                "slug": e.slug(),
                "audio_duration": e.audio_duration,
                "fps": e.fps,
                "timeline_html": f"entries/{e.slug()}/timeline.html",
                "timeline_txt": f"entries/{e.slug()}/timeline.txt",
                "timeline_json": f"entries/{e.slug()}/timeline.json",
                "text_preview": e.text[:160],
                "segment_count": len(t.segments),
                "duration_seconds": t.duration,
                "dominant_pose": _dominant_pose_of(t),
            }
            for e, t in entries_with_timelines
        ],
        "legend": legend,
    }
    index_path = gallery_path.parent / "gallery_index.json"
    index_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return index_path


# ─────────────────────────────────────────────────────────────────────────────
# Top-level builder + CLI plumbing
# ─────────────────────────────────────────────────────────────────────────────


def build_pose_gallery(
    entries: Sequence[GalleryEntry],
    output_dir: Path,
) -> dict[str, Path]:
    """Materialise the gallery artifacts in ``output_dir``.

    Returns a mapping ``{artifact_name: relative_path}`` so the CLI
    can print them and tests can assert on them.
    """
    if not entries:
        raise ValueError("build_pose_gallery requires at least one entry")
    duplicates = _find_duplicate_slugs(entries)
    if duplicates:
        raise ValueError(
            "duplicate (language, avatar_id) slugs detected — each card needs "
            "a unique subdirectory: " + ", ".join(sorted(duplicates))
        )
    output_dir = Path(output_dir)
    entries_dir = output_dir / "entries"
    entries_with_timelines: list[tuple[GalleryEntry, Timeline]] = []
    artifacts: dict[str, Path] = {}
    for entry in entries:
        subdir = entries_dir / entry.slug()
        _, _, html_path, timeline = _write_entry_artifact(entry, subdir)
        entries_with_timelines.append((entry, timeline))
        artifacts[f"entry:{entry.slug()}:timeline.html"] = html_path
    legend = _build_pose_legend(entries_with_timelines)
    gallery_path = _build_gallery_html(entries_with_timelines, output_dir, legend)
    artifacts["gallery.html"] = gallery_path
    index_path = _build_gallery_index(entries_with_timelines, legend, gallery_path)
    artifacts["gallery_index.json"] = index_path
    return artifacts


def _find_duplicate_slugs(entries: Sequence[GalleryEntry]) -> set[str]:
    """Return slugs that appear on more than one entry. Empty when unique."""
    seen: set[str] = set()
    dupes: set[str] = set()
    for entry in entries:
        slug = entry.slug()
        if slug in seen:
            dupes.add(slug)
        seen.add(slug)
    return dupes


def _resolve_entries(args: argparse.Namespace) -> list[GalleryEntry]:
    # argparse enforces --entries / --manifest mutual exclusion (SystemExit
    # on violation) so by the time we get here exactly one is set.
    if args.manifest is not None:
        return _load_yaml_entries(Path(args.manifest))
    if args.entries is not None:
        return _load_json_entries(args.entries)
    raise ValueError("no inputs: supply --manifest <yaml> or --entries '<json list>'")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an HTML pose gallery that embeds multiple per-(lang × avatar) timeline animations.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--manifest",
        type=Path,
        help="Path to a YAML manifest (a list of entries or {'entries': [...]}).",
    )
    group.add_argument(
        "--entries",
        type=str,
        help='Inline JSON list of entries: e.g. \'[{"language":"it","avatar_id":"alice","text":"Ciao","audio_duration":8.0}]\'',
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("captures/pose_gallery"),
        help="Where to write gallery.html + per-entry artifacts (default: captures/pose_gallery).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    entries = _resolve_entries(args)
    output_dir = Path(args.output_dir)
    artifacts = build_pose_gallery(entries, output_dir)
    print(
        f"[demo_pose_gallery] built gallery over {len(entries)} entries in {output_dir.resolve()}:"
    )
    for name, path in artifacts.items():
        print(f"  - {name:<48} -> {path}")
    print()
    print("Open gallery.html in your browser to compare the per-(lang × avatar) animations side-by-side.")
    return 0


__all__ = [
    "GalleryEntry",
    "build_pose_gallery",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
