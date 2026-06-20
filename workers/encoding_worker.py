"""Encoding worker — assembles rendered chunks into the final video.

Reads a chunk manifest produced by :class:`RenderVideo`, trims the
overlap between consecutive chunks, concatenates them, muxes the
original audio, and emits the final mp4 (or webm).

The overlap trimming solves the problem where chunk N and chunk N+1
share ``overlap_seconds`` of context that would otherwise appear twice
in the output. Each chunk is trimmed to its non-overlapping segment
before concatenation.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from src.core.config import Settings, get_settings
from src.core.logging import configure_logging


@dataclass(slots=True)
class EncodingWorker:
    settings: Settings
    output_dir: Path = field(default_factory=lambda: Path("./captures"))
    overlap_seconds: float = 0.5
    chunk_seconds: float = 4.0

    def encode(
        self,
        job_id: str,
        manifest_path: Path,
        *,
        audio_path: Optional[Path] = None,
        codec: str = "h264",
    ) -> Path:
        """Assemble chunks from *manifest_path* into a final video.

        The manifest format is one chunk per line:
        ``chunk_index|/absolute/path/to/chunk.mp4|duration_seconds``.
        Lines starting with ``#`` are comments.
        """
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise FileNotFoundError(
                "ffmpeg is required by the EncodingWorker; install it via your package manager."
            )
        entries = _parse_manifest(manifest_path)
        if not entries:
            raise RuntimeError(f"Manifest {manifest_path} contains no chunk entries.")

        out_path = self.output_dir / f"{job_id}.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if len(entries) == 1:
            # Single chunk — copy directly.
            shutil.copy2(entries[0][1], out_path)
        else:
            self._concat_with_trim(entries, ffmpeg, out_path, codec)

        # Mux original audio if provided.
        if audio_path is not None and audio_path.is_file():
            out_path = self._mux_audio(out_path, audio_path, ffmpeg, job_id)

        return out_path

    def _concat_with_trim(
        self,
        entries: List[Tuple[int, Path, float]],
        ffmpeg: str,
        out_path: Path,
        codec: str,
    ) -> None:
        """Concat chunks with overlap trimmed from non-first chunks.

        For each chunk N (N > 0), the first ``overlap_seconds`` are
        skipped so the overlap region from chunk N-1 is not repeated.
        Chunk 0 is kept in full.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            trimmed: List[Path] = []
            for i, (chunk_idx, chunk_path, duration) in enumerate(entries):
                if i == 0:
                    trimmed.append(chunk_path)
                else:
                    # Trim the first overlap_seconds from this chunk.
                    # Clamp to avoid trimming past chunk duration.
                    trim_start = min(self.overlap_seconds, max(duration * 0.4, 0.0))
                    if trim_start <= 0:
                        trimmed.append(chunk_path)
                        continue
                    trimmed_path = Path(tmpdir) / f"trimmed_{i:04d}.mp4"
                    cmd = [
                        ffmpeg, "-y", "-loglevel", "error",
                        "-ss", str(trim_start),
                        "-i", str(chunk_path),
                        "-c", "copy",
                        str(trimmed_path),
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        # If copy-mode trim fails, fall back to re-encode.
                        cmd[cmd.index("-c") + 1] = "libx264"
                        result = subprocess.run(cmd, capture_output=True, text=True)
                    trimmed.append(trimmed_path)

            # Write concat list and stitch.
            listfile = Path(tmpdir) / "concat.txt"
            with listfile.open("w", encoding="utf-8") as fh:
                for p in trimmed:
                    fh.write(f"file '{p.resolve().as_posix()}'\n")

            cmd = [
                ffmpeg, "-y", "-loglevel", "error",
                "-f", "concat", "-safe", "0",
                "-i", str(listfile),
                "-c:v", _resolve_codec(codec),
                "-pix_fmt", "yuv420p",
                str(out_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg concat failed: {result.stderr.strip()}")

    def _mux_audio(self, video_path: Path, audio_path: Path, ffmpeg: str, job_id: str) -> Path:
        """Mux the original audio track into the final video.

        Writes a temporary muxed file then atomically replaces the input.
        """
        muxed = self.output_dir / f"{job_id}.mp4"
        tmp = self.output_dir / f"{job_id}.mux.tmp.mp4"
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            str(tmp),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Audio mux failed — keep video without audio.
            tmp.unlink(missing_ok=True)
            return video_path
        # Replace the original with the muxed version.
        shutil.move(str(tmp), str(muxed))
        return muxed


def _parse_manifest(path: Path) -> List[Tuple[int, Path, float]]:
    """Parse a chunk manifest file."""
    entries: List[Tuple[int, Path, float]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) >= 3:
                entries.append((int(parts[0]), Path(parts[1]), float(parts[2])))
    return entries


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
    parser.add_argument("--audio", type=Path, default=None)
    parser.add_argument("--codec", default="h264")
    args = parser.parse_args()
    configure_logging()
    settings = get_settings()
    worker = EncodingWorker(settings=settings)
    out = worker.encode(args.job_id, args.manifest, audio_path=args.audio, codec=args.codec)
    print(out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(main())
