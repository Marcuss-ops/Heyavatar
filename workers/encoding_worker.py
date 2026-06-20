"""Encoding worker — appends rendered chunks into the final video.

Pulls `RenderResult` artefacts from the object store, concatenates them
with ffmpeg's concat demuxer, applies optional cross-fade on overlapping
sections, and emits the final mp4 (or webm).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.core.config import Settings, get_settings
from src.core.logging import configure_logging


@dataclass(slots=True)
class EncodingWorker:
    settings: Settings
    output_dir: Path = field(default_factory=lambda: Path("./captures"))

    def encode(self, job_id: str, manifest_path: Path, *, codec: str = "h264") -> Path:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise FileNotFoundError(
                "ffmpeg is required by the EncodingWorker; install it via your package manager."
            )
        out_path = self.output_dir / f"{job_id}.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
               "-i", str(manifest_path), "-c:v", _resolve_codec(codec), "-pix_fmt", "yuv420p",
               str(out_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {result.stderr.strip()}")
        return out_path


def _resolve_codec(name: str) -> str:
    if name == "h264":
        # Use NVIDIA NVENC when available, fall back to libx264 in CI / CPU.
        if shutil.which("nvidia-smi") is not None and shutil.which("ffmpeg"):
            return "h264_nvenc"
        return "libx264"
    if name == "vp9":
        return "libvpx-vp9"
    return name  # passthrough (e.g. 'av1')


def main() -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Run the encoding worker.")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--codec", default="h264")
    args = parser.parse_args()
    configure_logging()
    settings = get_settings()
    worker = EncodingWorker(settings=settings)
    out = worker.encode(args.job_id, args.manifest, codec=args.codec)
    print(out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(main())
