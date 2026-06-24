"""Demo CLI — wire the gesture pipeline end-to-end and emit animations.

This script is the "see it move" artifact the slim plan §6 timeline
section calls out: it plugs together the previously disconnected
pieces (``RuleBasedGesturePlanner`` → ``GestureRegistry`` →
``text_to_timeline`` → ASCII + SVG + HTML keyframe viewer) and writes
four files into ``--output-dir`` so an operator can both inspect the
data and *see* the planned motion without GPU or LivePortrait:

1. ``timeline.json`` — canonical Change 4 timeline shape
   (``src/motion/text_driven_timeline.Timeline``).
2. ``timeline.txt`` — ASCII bar chart with each segment labelled by
   pose and intensity. Pure stdlib.
3. ``timeline.svg`` — single static SVG stick figure at one pose
   with a labeled timeline underneath it. Standard `image/svg+xml`;
   openable in any browser.
4. ``timeline.html`` — CSS @keyframes animation of an SVG stick figure
   switching poses across the planned timeline. The browser plays
   the animation natively; no SMIL, no JS playback loop.

Run::

    python tools/avatar_assets/demo_gesture_timeline.py \\
      --text "Benvenuti! Oggi parliamo di tre differenze importanti." \\
      --audio-duration 8.5 \\
      --output-dir captures/demo_gesture

The script is import-safe on a stock Python install — only
``pydantic`` and ``pyyaml`` are touched, both of which are in
``pyproject.toml``. No numpy, cv2, mediapipe, torch, or LivePortrait
imports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from src.motion.text_driven_timeline import Timeline, TimelineSegment, text_to_timeline


# ─────────────────────────────────────────────────────────────────────────────
# Pose → animated arm joint angles (visualization only)
# ─────────────────────────────────────────────────────────────────────────────
#
# Each pose pivots the arms at the shoulders. The numbers are
# screen-space rotation degrees in a 200×200 viewBox (left arm pivots
# around the LEFT shoulder, positive rotation brings the hand outward,
# negative pulls it inward and up toward the head).
#
# The mapping lives next to the animator because it is purely a
# presentation concern; it never affects the canonical JSON timeline
# emitted for the real render path.

_POSE_ARMS: dict[str, tuple[float, float]] = {
    "neutral_desk": (0.0, 0.0),
    "right_hand_up": (15.0, -110.0),
    "left_hand_up": (-110.0, 15.0),
    "both_hands_open": (-55.0, 55.0),
    "right_hand_rising": (10.0, -45.0),
    "left_hand_rising": (-45.0, 10.0),
    "right_hand_lowering": (5.0, -25.0),
    "left_hand_lowering": (-25.0, 5.0),
}


def _arms_for_pose(pose_id: str) -> tuple[float, float]:
    return _POSE_ARMS.get(pose_id, (0.0, 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# ASCII bar chart
# ─────────────────────────────────────────────────────────────────────────────

_BAR_WIDTH = 64
_TICKS_PER_SECOND = 4


def _format_ascii(timeline: Timeline) -> str:
    """Render the timeline as a 64-column ASCII bar chart per second.

    Each row shows one second of the timeline; characters represent the
    dominant pose at that second. A header row prints the same info as
    a single line per second for quick scanning.
    """
    if timeline.duration <= 0.0:
        return "(empty timeline — audio_duration is zero)\n"

    bars: list[str] = []
    total_seconds = max(1, int(round(timeline.duration)))
    for sec in range(total_seconds):
        t = sec + 0.5  # sample at the centre of each second
        seg = _segment_at(timeline.segments, t)
        bar_ch = _bar_char(seg)
        bars.append(f"t={sec:3d}s | {bar_ch * _BAR_WIDTH} | {seg.gesture_id:<22s}")

    title = f"heyavatar gesture timeline (duration={timeline.duration:.2f}s, fps={timeline.fps})"
    footer = "\n. = idle  ~ = gesture-light  # = gesture-medium  | = gesture-strong\n"
    return f"{title}\n" + "\n".join(bars) + footer


def _segment_at(segments: tuple[TimelineSegment, ...], t: float) -> TimelineSegment:
    for seg in segments:
        if seg.start <= t <= seg.end:
            return seg
    return segments[-1]


def _bar_char(seg: TimelineSegment) -> str:
    # ASCII-safe fallback bars so the chart prints cleanly on Windows
    # cp1252 consoles as well as POSIX UTF-8. The richer UTF-8 set
    # (·, ▓, █) lives in the SVG/HTML viewers, not the console.
    if seg.kind == "idle":
        return "."
    if seg.intensity >= 0.85:
        return "|"
    if seg.intensity >= 0.6:
        return "#"
    if seg.intensity >= 0.3:
        return "~"
    return "."


# ─────────────────────────────────────────────────────────────────────────────
# Stick figure renderer (shared by single-frame SVG + keyframe HTML)
# ─────────────────────────────────────────────────────────────────────────────


_STICK_FIGURE_SVG = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="320" height="320" role="img" aria-label="Avatar stick figure pose">
  <style>
    .ground {{ stroke:#888; stroke-width:1; }}
    .body   {{ stroke:#222; stroke-width:4; stroke-linecap:round; }}
    .head   {{ fill:#f3d6b1; stroke:#222; stroke-width:2; }}
    .left-arm  {{ stroke:#2473ff; stroke-width:4; stroke-linecap:round; }}
    .right-arm {{ stroke:#ff5722; stroke-width:4; stroke-linecap:round; }}
  </style>
  <line class="ground" x1="20" y1="180" x2="180" y2="180"/>
  <line class="body" x1="100" y1="60" x2="100" y2="150"/>
  <line class="body" x1="100" y1="150" x2="100" y2="175"/>
  <line class="body" x1="100" y1="150" x2="80" y2="175"/>
  <line class="body" x1="100" y1="150" x2="120" y2="175"/>
  <circle class="head" cx="100" cy="40" r="22"/>
  <g class="left-arm" style="transform-origin: 90px 60px; transform: rotate(0deg);">
    <line x1="90" y1="60" x2="60" y2="110"/>
  </g>
  <g class="right-arm" style="transform-origin: 110px 60px; transform: rotate(0deg);">
    <line x1="110" y1="60" x2="140" y2="110"/>
  </g>
</svg>
"""


def _static_svg(timeline: Timeline) -> str:
    """Build a single SVG with the avatar in the dominant pose and a
    labelled timeline underneath."""
    dominant = _dominant_segment(timeline.segments)
    left_deg, right_deg = _arms_for_pose(dominant.pose_id)
    figure = _STICK_FIGURE_SVG.replace(
        'transform: rotate(0deg);',
        f'transform: rotate({left_deg:.1f}deg);',
        1,
    ).replace(
        'transform: rotate(0deg);',
        f'transform: rotate({right_deg:.1f}deg);',
        1,
    )
    timeline_strip = _timeline_strip_svg(timeline)
    caption = f"<h2>dominant pose: {dominant.pose_id}</h2>"
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 380" width="600" height="380" role="img" aria-label="Heyavatar gesture timeline preview">
  <style>
    .label {{ font: 14px sans-serif; fill: #222; }}
    .tick  {{ stroke: #888; stroke-width: 1; }}
    .seg-idle  {{ fill: #d8e1ef; stroke: #5a78a8; stroke-width: 1; }}
    .seg-gesture-light  {{ fill: #b3d8ff; stroke: #2473ff; }}
    .seg-gesture-med    {{ fill: #79b6ff; stroke: #2473ff; }}
    .seg-gesture-strong {{ fill: #2473ff; stroke: #003e80; }}
  </style>
  <g transform="translate(20,20)">{figure}</g>
  <text class="label" x="340" y="40">{caption.strip().strip('<>h2/')}</text>
  {timeline_strip}
</svg>"""


def _timeline_strip_svg(timeline: Timeline) -> str:
    """Render the timeline as a single horizontal stacked bar."""
    if timeline.duration <= 0.0:
        return '<text x="20" y="320" class="label">(empty timeline)</text>'
    parts: list[str] = []
    for seg in timeline.segments:
        x = 20 + (seg.start / timeline.duration) * 560
        w = max(2.0, ((seg.end - seg.start) / timeline.duration) * 560)
        cls = "seg-idle"
        if seg.kind == "gesture":
            if seg.intensity >= 0.85:
                cls = "seg-gesture-strong"
            elif seg.intensity >= 0.6:
                cls = "seg-gesture-med"
            else:
                cls = "seg-gesture-light"
        parts.append(
            f'<rect class="{cls}" x="{x:.2f}" y="290" width="{w:.2f}" height="22" rx="3"/>'
        )
        parts.append(
            f'<text class="label" x="{x + 4:.2f}" y="306">{seg.gesture_id}</text>'
        )
    n = max(1, int(round(timeline.duration)))
    ticks = '\n'.join(
        f'<line class="tick" x1="{20 + tick * 560 / n:.2f}" y1="280" x2="{20 + tick * 560 / n:.2f}" y2="285"/>'
        f'<text class="label" x="{20 + tick * 560 / n - 6:.2f}" y="278">{tick}s</text>'
        for tick in range(n + 1)
    )
    return "\n".join(parts) + "\n" + ticks


def _dominant_segment(segments: tuple[TimelineSegment, ...]) -> TimelineSegment:
    if not segments:
        return TimelineSegment(
            kind="idle", start=0.0, end=0.0, gesture_id="idle_small", pose_id="neutral_desk"
        )
    return max(segments, key=lambda s: (s.end - s.start) * max(s.intensity, 0.1))


# ─────────────────────────────────────────────────────────────────────────────
# HTML keyframe animation
# ─────────────────────────────────────────────────────────────────────────────


def _format_html(timeline: Timeline) -> str:
    """Build a self-contained HTML file with CSS-keyframed SVG animation.

    The mood: open in a browser → the stick figure goes through every
    pose across the audio_duration, hands sweeping through the
    timeline. No SMIL, no JS — just CSS.
    """
    if timeline.duration <= 0.0:
        return "<h1>empty timeline</h1>"

    left_keyframes: list[tuple[float, float]] = []
    right_keyframes: list[tuple[float, float]] = []
    label_keyframes: list[tuple[float, str]] = []

    # Always start at 0s with the first segment's pose.
    left0, right0 = _arms_for_pose(timeline.segments[0].pose_id)
    left_keyframes.append((0.0, left0))
    right_keyframes.append((0.0, right0))
    label_keyframes.append((0.0, timeline.segments[0].gesture_id))

    for seg in timeline.segments:
        pct_start = (seg.start / timeline.duration) * 100.0
        # Interpolate half-way through the segment to the apex pose so
        # the motion looks like a stroke rather than a teleport.
        pct_apex = ((seg.start + (seg.end - seg.start) * 0.4) / timeline.duration) * 100.0
        left_apex, right_apex = _arms_for_pose(seg.pose_id)
        # Each segment gets two keyframes (apex pose + return pose).
        left_keyframes.append((max(pct_apex, pct_start + 0.1), left_apex))
        right_keyframes.append((max(pct_apex, pct_start + 0.1), right_apex))
        label_keyframes.append((max(pct_apex, pct_start + 0.1), seg.gesture_id))
        # The animation continues to hold the apex through the
        # remaining 60% of the segment so the pose reads on screen.
        pct_hold = ((seg.start + (seg.end - seg.start) * 0.9) / timeline.duration) * 100.0
        left_keyframes.append((pct_hold, left_apex))
        right_keyframes.append((pct_hold, right_apex))
        label_keyframes.append((pct_hold, seg.gesture_id))
    # End exactly at 100% so the loop closes cleanly.
    last_left, last_right = _arms_for_pose(timeline.segments[-1].pose_id)
    left_keyframes.append((100.0, last_left))
    right_keyframes.append((100.0, last_right))

    left_css = _keyframe_block("left-arm", left_keyframes)
    right_css = _keyframe_block("right-arm", right_keyframes)

    segment_table = _segment_summary_table(timeline)
    timeline_strip = _timeline_strip_svg(timeline)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>heyavatar gesture timeline preview</title>
<style>
  :root {{
    --bg: #f7f7f4;
    --fg: #222;
    --accent: #2473ff;
  }}
  body {{ font: 15px/1.4 system-ui, sans-serif; background: var(--bg); color: var(--fg); margin: 24px; }}
  h1 {{ margin: 0 0 8px 0; }}
  .meta {{ color: #555; font-size: 13px; margin-bottom: 14px; }}
  .stage {{ display: flex; gap: 24px; align-items: flex-start; flex-wrap: wrap; }}
  .figure svg {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; }}
  .label-pill {{
    display: inline-block; padding: 6px 10px; border: 1px solid #ccc; border-radius: 999px;
    font-size: 13px; background: #fff;
    animation: pill 1s ease-in-out infinite alternate;
  }}
  .figure {{ display: flex; flex-direction: column; align-items: center; }}
  svg .left-arm  {{ transform-origin: 90px 60px;  animation: left-arm  $(duration)s linear infinite; }}
  svg .right-arm {{ transform-origin: 110px 60px; animation: right-arm $(duration)s linear infinite; }}
  svg .head-body {{ fill: #f3d6b1; stroke: #222; stroke-width: 2; }}
  {left_css}
  {right_css}
  @keyframes pill {{ 0% {{ opacity: 0.9; }} 100% {{ opacity: 1; }} }}
  table {{ border-collapse: collapse; min-width: 540px; }}
  th, td {{ padding: 6px 10px; border-bottom: 1px solid #eee; text-align: left; }}
  th {{ background: #eee; }}
  .seg-idle  {{ background: #f0f4fa; }}
</style>
</head>
<body>
  <h1>heyavatar gesture timeline preview</h1>
  <div class="meta">
    duration: <b>{timeline.duration:.2f}s</b> · fps: <b>{timeline.fps}</b> ·
    segments: <b>{len(timeline.segments)}</b> · pose classes: {(len(set(s.pose_id for s in timeline.segments)))}
  </div>
  <div class="stage">
    <div class="figure">
      {_STICK_FIGURE_SVG}
      <div class="label-pill">gesture: <b>$(first_gesture)</b></div>
    </div>
    <div>
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 380" width="600" height="320" style="background:#fff;border:1px solid #ddd;border-radius:8px;">
        <style>
          .label {{ font: 14px sans-serif; fill: #222; }}
          .tick  {{ stroke: #888; stroke-width: 1; }}
          .seg-idle  {{ fill: #d8e1ef; stroke: #5a78a8; stroke-width: 1; }}
          .seg-gesture-light  {{ fill: #b3d8ff; stroke: #2473ff; }}
          .seg-gesture-med    {{ fill: #79b6ff; stroke: #2473ff; }}
          .seg-gesture-strong {{ fill: #2473ff; stroke: #003e80; }}
        </style>
        {timeline_strip}
      </svg>
    </div>
  </div>
  <h2>segments</h2>
  {segment_table}
</body>
</html>"""


def _keyframe_block(name: str, keyframes: list[tuple[float, float]]) -> str:
    """Render the CSS for one @keyframes rule given a list of (pct,
    rotation_deg) tuples. Coalesces consecutive duplicates so the
    browser doesn't fight itself on identical angles."""
    cleaned: list[tuple[float, float]] = []
    for pct, deg in sorted(keyframes, key=lambda x: x[0]):
        if cleaned and abs(cleaned[-1][1] - deg) < 1e-3 and abs(cleaned[-1][0] - pct) < 0.5:
            continue
        cleaned.append((pct, deg))
    rules = "\n".join(f"  {pct:.1f}% {{ transform: rotate({deg:.1f}deg); }}" for pct, deg in cleaned)
    return f"@keyframes {name} {{\n{rules}\n}}"


def _segment_summary_table(timeline: Timeline) -> str:
    head = (
        "<tr>"
        "<th>#</th><th>kind</th><th>start</th><th>end</th>"
        "<th>duration</th><th>gesture</th><th>pose</th>"
        "<th>intensity</th><th>anchor</th>"
        "</tr>"
    )
    rows = "\n".join(
        f"<tr class='seg-{seg.kind}'>"
        f"<td>{idx}</td><td>{seg.kind}</td>"
        f"<td>{seg.start:.2f}s</td><td>{seg.end:.2f}s</td><td>{seg.end - seg.start:.2f}s</td>"
        f"<td>{seg.gesture_id}</td><td>{seg.pose_id}</td>"
        f"<td>{seg.intensity:.2f}</td><td>{seg.anchor_word or '-'}</td>"
        "</tr>"
        for idx, seg in enumerate(timeline.segments)
    )
    return f"<table>{head}{rows}</table>"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_segment_summary(segments: Any) -> Mapping[str, Any]:
    """Reduce the timeline for the JSON sidecar we write alongside."""
    if not segments:
        return {"pose_classes": [], "gesture_count": 0, "idle_count": 0}
    gestures = sum(1 for s in segments if s.kind == "gesture")
    idles = sum(1 for s in segments if s.kind == "idle")
    pose_classes = sorted({s.pose_id for s in segments})
    return {
        "pose_classes": pose_classes,
        "gesture_count": gestures,
        "idle_count": idles,
        "max_intensity": max((s.intensity for s in segments), default=0.0),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Wire the gesture pipeline end-to-end and emit ASCII + SVG + HTML animation preview.",
    )
    parser.add_argument(
        "--text",
        required=True,
        help="The script the avatar is going to speak. Italian + English keywords supported.",
    )
    parser.add_argument(
        "--audio-duration",
        type=float,
        default=8.0,
        help="Estimated audio length of the clip in seconds (default: 8.0).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("captures/demo_gesture"),
        help="Directory to write timeline.json / .txt / .svg / .html into.",
    )
    parser.add_argument(
        "--avatar-id",
        default="default",
        help="Avatar id forwarded to the planner (default: 'default').",
    )
    parser.add_argument(
        "--language",
        default="it",
        help="Language forwarded to the planner ('it' or 'en' — default 'it').",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Output framerate (default 25, matches registry/models.yaml::standard).",
    )
    return parser


def _write_outputs(timeline: Timeline, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "timeline.json"
    txt_path = output_dir / "timeline.txt"
    svg_path = output_dir / "timeline.svg"
    html_path = output_dir / "timeline.html"

    payload = timeline.to_dict()
    summary = _resolve_segment_summary(timeline.segments)
    payload["summary"] = summary

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    txt_path.write_text(_format_ascii(timeline), encoding="utf-8")
    svg_path.write_text(_static_svg(timeline), encoding="utf-8")

    html = _format_html(timeline)
    html = html.replace("$(duration)", f"{timeline.duration:.2f}")
    html = html.replace("$(first_gesture)", timeline.segments[0].gesture_id if timeline.segments else "-")
    html_path.write_text(html, encoding="utf-8")

    return {
        "timeline.json": json_path,
        "timeline.txt": txt_path,
        "timeline.svg": svg_path,
        "timeline.html": html_path,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    timeline = text_to_timeline(
        text=args.text,
        audio_duration=args.audio_duration,
        avatar_id=args.avatar_id,
        language=args.language,
        fps=args.fps,
    )
    outputs = _write_outputs(timeline, args.output_dir)
    print(
        f"[demo_gesture_timeline] planned {len(timeline.segments)} segments "
        f"({timeline.duration:.2f}s @ {timeline.fps}fps) under {args.output_dir.resolve()}:"
    )
    for name, path in outputs.items():
        print(f"  - {name:<13} -> {path}")
    print()
    print("Open the HTML file in your browser to see the animation play.")
    print()
    print("ASCII preview:")
    print(_format_ascii(timeline))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
