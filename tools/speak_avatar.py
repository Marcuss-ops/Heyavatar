"""Generate a speaking avatar video from text."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from src.application.speak_avatar import speak_avatar_sync


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
        default=Path("assets/green_screen_avatar.jpg"),
        help="Avatar source image. Default: assets/green_screen_avatar.jpg",
    )
    parser.add_argument("--avatar-id", default="my_avatar", help="Identity key to use.")
    parser.add_argument(
        "--engine",
        default="liveportrait-human-v1",
        help="Engine id or canonical name.",
    )
    parser.add_argument("--language", default="it")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--capture-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Optional pre-recorded audio instead of synthesizing speech.",
    )
    parser.add_argument(
        "--audio-bridge-backend",
        choices=("dsp", "neural", "auto"),
        default="dsp",
        help="Audio bridge backend used by the render pipeline.",
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

    result = speak_avatar_sync(
        args.text,
        voice=args.voice,
        avatar_id=args.avatar_id,
        source_image=args.source_image,
        engine=args.engine,
        language=args.language,
        fps=args.fps,
        capture_root=args.capture_root,
        output_path=args.output,
        audio=args.audio,
        audio_bridge_backend=args.audio_bridge_backend,
    )

    if result.video_path is None:
        raise SystemExit(result.error or "render failed")

    print(f"final mp4: {result.video_path}")
    print(f"audio wav:  {result.audio_path}")
    print(f"status:     {result.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
