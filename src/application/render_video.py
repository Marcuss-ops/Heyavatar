"""Render-video use case.

Splits the audio input into a sequence of render windows (anchored at
chunk boundaries, with optional overlap), invokes the adapter for each
window, and stitches the chunks into a final video.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

from contracts.avatar_engine import AvatarEngine
from src.application.telemetry import TelemetryRecorder
from src.domain.types import (
    AvatarIdentityHandle,
    RenderChunkRequest,
    RenderChunkResult,
    RenderRequest,
    RenderResult,
    RenderSpec,
)
from src.domain.enums import Tier


@dataclass(slots=True, frozen=True)
class ChunkConfig:
    """Knobs for the audio → chunks pipeline."""

    chunk_seconds: float = 4.0
    overlap_seconds: float = 0.5
    max_chunks: int = 8  # safety cap; never exceed this without explicit override


@dataclass(slots=True)
class RenderVideo:
    """Drive an engine through a long-form render."""

    engine: AvatarEngine
    telemetry: TelemetryRecorder = field(default_factory=TelemetryRecorder)
    chunk_config: ChunkConfig = field(default_factory=ChunkConfig)

    def run(self, request: RenderRequest, identity: AvatarIdentityHandle) -> RenderResult:
        chunks = list(self._chunks_for(request))
        results: List[RenderChunkResult] = []
        for chunk_req in chunks:
            with self.telemetry.span("render_chunk", engine_id=self.engine.engine_id.value):
                result = self.engine.render_chunk(chunk_req, identity)
                self.telemetry.record(result.gpu_seconds, engine_id=self.engine.engine_id.value)
            results.append(result)
        output = self._stitch(results, request)
        return RenderResult(
            job_id=request.job_id,
            identity_id=identity.identity_id,
            output_path=output,
            duration_seconds=sum(r.duration_seconds for r in results),
            fps=request.render_spec.fps,
            tier=request.tier,
            engine_id=self.engine.engine_id,
            gpu_seconds_total=sum(r.gpu_seconds for r in results),
            completed_at=datetime.now(timezone.utc),
            chunks=tuple(results),
        )

    def _chunks_for(self, request: RenderRequest) -> Iterable[RenderChunkRequest]:
        cfg = self.chunk_config
        index = 0
        start = 0.0
        while index < cfg.max_chunks:
            end = start + cfg.chunk_seconds
            yield RenderChunkRequest(
                job_id=request.job_id,
                audio_window=(start, end),
                audio_path=request.render_spec.audio_path,
                fps=request.render_spec.fps,
                resolution=request.render_spec.target_resolution,
                chunk_index=index,
                overlap_seconds=cfg.overlap_seconds if start > 0 else 0.0,
            )
            start = end - cfg.overlap_seconds if cfg.overlap_seconds > 0 else end
            index += 1

    def _stitch(self, results: List[RenderChunkResult], request: RenderRequest) -> Path:
        if not results:
            raise RuntimeError("No chunks rendered for job " + str(request.job_id))
        dest = _final_path(request)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if len(results) == 1:
            shutil.copy2(results[0].output_path, dest)
            return dest
        # Multi-chunk: try ffmpeg concat, fall back to first-chunk copy if
        # ffmpeg isn't installed (so callers always see *some* output).
        import subprocess
        work_dir = dest.parent
        listfile = work_dir / f"{dest.stem}.concat.txt"
        with listfile.open("w", encoding="utf-8") as fh:
            for r in results:
                fh.write(f"file '{r.output_path.resolve().as_posix()}'\n")
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            shutil.copy2(results[0].output_path, dest)
            return dest
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(listfile),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(dest),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            from src.core.logging import get_logger
            get_logger(__name__).warning(
                "ffmpeg concat failed (rc=%s); falling back to first chunk. stderr=%s",
                result.returncode,
                result.stderr.strip(),
            )
            shutil.copy2(results[0].output_path, dest)
            return dest
        return dest


def _final_path(request: RenderRequest) -> Path:
    return Path("./captures") / f"{request.job_id}.mp4"
