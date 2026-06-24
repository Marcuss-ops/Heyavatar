"""``tools/run_cached_avatar.py`` — one-shot CLI: text + audio → 1 final mp4.

This is the operational entry point promised at the end of last turn:
previously the demo CLI emitted 4 artifacts (timeline.json +
timeline.txt + timeline.svg + timeline.html) on a small Italian
script. This script closes that gap by producing one real mp4 the
user can hit play on.

It is a thin wrapper around
:func:`src.application.run_cached_avatar_from_text.run_cached_avatar_from_text`:

1. ``HEYAVATAR_MOCK_ENGINE=1`` is set as a default so the CLI runs
   on a CPU box without exhausting the operator's patience; flip to
   ``HEYAVATAR_MOCK_ENGINE=0`` for real-mode runs on a CUDA host.
2. ``--engine musetalk-v1`` picks the default engine adapter
   (``MUSE_TALK``) — the production-safe pick for face-region-only
   rendering. ``--engine liveportrait-human-v1`` is the alternative
   for full-body / higher fidelity once the upstream clone is ready.
3. ``--mode timeline`` iterates the planned timeline segment by
   segment, slice + render + concat. ``--mode dominant`` collapses
   to one body pose covering the whole audio.
4. The CLI owns engine lifecycle: load once, run, unload in finally.

Example::

    HEYAVATAR_MOCK_ENGINE=1 python tools/run_cached_avatar.py \
      --text "Ciao a tutti! Oggi parliamo di tre differenze molto importanti." \
      --audio samples/speech_30s.wav \
      --avatar-id alice \
      --output captures/text_demo.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path
from typing import Optional

from providers import get_provider
from src.application.run_cached_avatar_from_text import (
    FromTextRunResult,
    run_cached_avatar_from_text,
)
from src.core.config import get_settings
from src.domain.enums import EngineId


# Canonical names (from registry/models.yaml) → EngineId enum value.
# Strings are dash-separated by registry convention; the enum values
# match them but the enum NAMES use underscores. Hardcoding the
# cross-walk here keeps the CLI permissive and the registry source
# of truth.
_CANONICAL_TO_ENGINE: dict[str, EngineId] = {
    "musetalk-v1": EngineId.MUSE_TALK,
    "liveportrait-human-v1": EngineId.LIVE_PORTRAIT,
    "echomimic-v1": EngineId.ECHO_MIMIC,
}


def _resolve_engine_id(name: Optional[str]) -> EngineId:
    """Map a CLI string to an :class:`EngineId`, falling back to MUSE_TALK.

    Accepts both canonical names (``musetalk-v1``) and bare enum
    values (``musetalk-v1`` is the canonical — the enum value
    matches, the enum name has underscores). Unknown strings raise
    :class:`ValueError` so the operator gets a clean error instead
    of a torch import crash later.
    """
    if name is None:
        return EngineId.MUSE_TALK
    cleaned = name.strip().lower()
    if cleaned in _CANONICAL_TO_ENGINE:
        return _CANONICAL_TO_ENGINE[cleaned]
    try:
        return EngineId(cleaned)
    except ValueError as exc:
        raise ValueError(
            f"unknown engine {name!r}; valid choices: "
            f"{sorted(_CANONICAL_TO_ENGINE.keys())}"
        ) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run avatar render from text + audio. Produces 1 final mp4.",
    )
    parser.add_argument("--text", required=True, help="The script the avatar speaks.")
    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="Source WAV / mp3 driving the lip-sync.",
    )
    parser.add_argument(
        "--avatar-id",
        default="default",
        help="Identity key + body-template subdirectory (default: 'default').",
    )
    parser.add_argument(
        "--identity-pack",
        type=Path,
        default=None,
        help="Optional pre-built avatar pack (.tar) — short-circuits compile.",
    )
    parser.add_argument(
        "--source-image",
        type=Path,
        default=None,
        help="Optional source PNG used to compile a fresh pack when cache misses.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="musetalk-v1",
        help=(
            "Engine canonical name (musetalk-v1 | liveportrait-human-v1 | "
            "echomimic-v1). Default: musetalk-v1."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("timeline", "dominant"),
        default="timeline",
        help="timeline = iterate every gesture segment; dominant = single pose.",
    )
    parser.add_argument("--language", default="it", help="Planner language (it/en).")
    parser.add_argument("--fps", type=int, default=25, help="Output framerate.")
    parser.add_argument(
        "--motion-style",
        type=str,
        default=None,
        help="Optional motion profile: subtle | natural | balanced | expressive.",
    )
    parser.add_argument(
        "--body-templates-dir",
        type=Path,
        default=Path("body_templates"),
        help="Base dir for cached body templates (default: body_templates).",
    )
    parser.add_argument(
        "--capture-root",
        type=Path,
        default=None,
        help="Override the capture root (default: settings.capture_dir).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Final mp4 path (default: captures/<job_id>.mp4).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    # Default mock-mode to 1 so a CLI run on a lean box is not a 5-minute
    # VRAM load followed by a hard crash. Override with
    # ``HEYAVATAR_MOCK_ENGINE=0`` to flip into real-mode.
    os.environ.setdefault("HEYAVATAR_MOCK_ENGINE", "1")

    engine_id = _resolve_engine_id(args.engine)
    settings = get_settings()
    capture_root = args.capture_root or settings.capture_dir
    output_path = args.output or capture_root / f"run-{uuid.uuid4().hex[:10]}.mp4"
    job_id = f"run-{uuid.uuid4().hex[:10]}"

    engine = get_provider(engine_id)
    # The CLI owns the engine lifecycle so VRAM + CUDA contexts are
    # reclaimed deterministically after the run, even if the
    # library function raises mid-loop.
    engine.load()
    try:
        result: FromTextRunResult = run_cached_avatar_from_text(
            text=args.text,
            audio_path=args.audio,
            output_path=output_path,
            avatar_id=args.avatar_id,
            engine=engine,
            language=args.language,
            fps=args.fps,
            mode=args.mode,
            body_templates_dir=args.body_templates_dir,
            capture_root=capture_root,
            source_image=args.source_image,
            identity_pack_path=args.identity_pack,
            pack_root=settings.pack_dir,
            job_id=job_id,
            motion_style=args.motion_style,
        )
    finally:
        engine.unload()

    print(f"[run_cached_avatar] mode={result.mode} job_id={result.job_id}")
    print(f"[run_cached_avatar] planned {len(result.timeline.segments)} timeline segments")
    print(
        f"[run_cached_avatar] rendered {result.segment_count} job chunks "
        f"({result.render_seconds_total:.2f}s total)"
    )
    if result.skipped_gestures:
        print(
            f"[run_cached_avatar] WARN: gestures without body templates "
            f"(fell back to idle_small): {', '.join(result.skipped_gestures)}"
        )
    print(f"[run_cached_avatar] final mp4: {result.output_path}")
    print(f"[run_cached_avatar] chunk manifest: {result.chunk_manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main"]
