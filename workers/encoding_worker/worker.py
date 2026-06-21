"""``EncodingWorker`` — chunk-list assembler + overlap trimmer + audio muxer.

The worker reads a chunk manifest file written by :class:`RenderVideo`,
:func:`_parse_manifest` parses the entries, then for each non-first
chunk the leading ``overlap_seconds`` are stripped from the source mp4
so consecutive chunks don't double-count shared context. ffmpeg then
re-stitches the trimmed segments and (optionally) muxes the original
audio back into the final video.

See :mod:`workers.encoding_worker.manifest` and
:mod:`workers.encoding_worker.codec` for the lower-level helpers.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from src.core.config import Settings

from workers.encoding_worker.codec import _resolve_codec
from workers.encoding_worker.manifest import _parse_manifest


@dataclass(slots=True)
class EncodingWorker:
    """Assembles rendered chunks into the final video file."""

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

    def _mux_audio(
        self, video_path: Path, audio_path: Path, ffmpeg: str, job_id: str
    ) -> Path:
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
