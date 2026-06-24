"""Generate a speaking avatar video from text.

This tool stitches together speech synthesis and the existing avatar
render pipeline, using the cleaned avatar base asset as the source image
by default.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

from providers import get_provider
from src.application.run_cached_avatar_from_text import run_cached_avatar_from_text
from src.application.speech_synthesis import synthesize_speech
from src.core.config import get_settings
from src.domain.enums import EngineId


_CANONICAL_TO_ENGINE: dict[str, EngineId] = {
    "musetalk-v1": EngineId.MUSE_TALK,
    "liveportrait-human-v1": EngineId.LIVE_PORTRAIT,
    "echomimic-v1": EngineId.ECHO_MIMIC,
}


def _resolve_engine_id(name: str) -> EngineId:
    cleaned = name.strip().lower()
    if cleaned in _CANONICAL_TO_ENGINE:
        return _CANONICAL_TO_ENGINE[cleaned]
    return EngineId(cleaned)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthesize speech and render a talking avatar.")
    parser.add_argument("--text", required=True, help="What the avatar should say.")
    parser.add_argument(
        "--voice",
        default="it-IT-DiegoNeural",
        help="TTS voice name. Default: it-IT-DiegoNeural",
    )
    parser.add_argument(
        "--source-image",
        type=Path,
        default=Path("assets/avatar_base_cutout.svg"),
        help="Avatar base image. Default: assets/avatar_base_cutout.svg",
    )
    parser.add_argument("--avatar-id", default="my_avatar", help="Body-template key to use.")
    parser.add_argument(
        "--engine",
        default="liveportrait-human-v1",
        help="Engine id or canonical name.",
    )
    parser.add_argument("--mode", choices=("timeline", "dominant"), default="timeline")
    parser.add_argument("--language", default="it")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--motion-style", default=None)
    parser.add_argument(
        "--audio-bridge-backend",
        choices=("dsp", "neural"),
        default="dsp",
        help="Audio bridge backend used by the render pipeline.",
    )
    parser.add_argument("--body-templates-dir", type=Path, default=Path("body_templates"))
    parser.add_argument("--capture-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Optional pre-recorded audio instead of synthesizing speech.",
    )
    parser.add_argument(
        "--skip-checkpoint-verify",
        action="store_true",
        help="Skip checkpoint SHA256 verification during first-time setup.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    os.environ.setdefault("HEYAVATAR_MOCK_ENGINE", "0")
    if args.skip_checkpoint_verify:
        os.environ["HEYAVATAR_SKIP_SHA256_VERIFY"] = "1"
    os.environ["HEYAVATAR_AUDIO_BRIDGE_BACKEND"] = args.audio_bridge_backend

    settings = get_settings()
    capture_root = args.capture_root or settings.capture_dir
    output_path = args.output or (capture_root / f"talk-{uuid.uuid4().hex[:10]}.mp4")
    audio_path = args.audio or (capture_root / f"talk-{uuid.uuid4().hex[:10]}.wav")

    if args.audio is None:
        synthesize_speech(args.text, audio_path, voice=args.voice)

    engine_id = _resolve_engine_id(args.engine)
    engine = get_provider(engine_id)
    engine.load()
    try:
        result = run_cached_avatar_from_text(
            text=args.text,
            audio_path=audio_path,
            output_path=output_path,
            avatar_id=args.avatar_id,
            engine=engine,
            language=args.language,
            fps=args.fps,
            mode=args.mode,
            body_templates_dir=args.body_templates_dir,
            capture_root=capture_root,
            source_image=args.source_image,
            pack_root=settings.pack_dir,
            motion_style=args.motion_style,
        )
    finally:
        engine.unload()

    print(f"final mp4: {result.output_path}")
    print(f"audio wav:  {audio_path}")
    print(f"job id:     {result.job_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
