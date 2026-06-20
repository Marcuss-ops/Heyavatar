"""Render-video use case.

Splits the audio input into a sequence of render windows (anchored at
chunk boundaries, with optional overlap), invokes the adapter for each
window with **per-chunk retry**, publishes GPU-seconds telemetry per
chunk (so failed jobs don't lose accounting), and writes a manifest for
the :class:`EncodingWorker` to assemble the final video.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

from contracts.avatar_engine import AvatarEngine
from src.application.telemetry import TelemetryRecorder
from src.core.logging import get_logger
from src.domain.types import (
    AvatarIdentityHandle,
    RenderChunkRequest,
    RenderChunkResult,
    RenderRequest,
    RenderResult,
    RenderSpec,
)
from src.domain.enums import Tier

LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Audio probing
# ---------------------------------------------------------------------------


def _probe_audio_duration(audio_path: Path) -> float:
    """Return the duration of an audio file in seconds via ffprobe.

    Returns 0.0 if ffprobe is not available or the file cannot be read,
    so callers fall back gracefully.
    """
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return 0.0
    try:
        result = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except (ValueError, subprocess.TimeoutExpired, OSError):
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Chunk configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class ChunkConfig:
    """Knobs for the audio → chunks pipeline."""

    chunk_seconds: float = 4.0
    overlap_seconds: float = 0.5
    max_chunks: int = 200  # safety cap based on duration, raised from 8
    chunk_retry_max: int = 3  # retry a single failed chunk up to this many times
    chunk_retry_delay_seconds: float = 0.5  # sleep between retries


# ---------------------------------------------------------------------------
# RenderVideo
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RenderVideo:
    """Drive an engine through a long-form render.

    Each chunk is rendered independently with retry on failure. GPU
    telemetry is published per-chunk so partial-job costs are visible.
    """

    engine: AvatarEngine
    telemetry: TelemetryRecorder = field(default_factory=TelemetryRecorder)
    chunk_config: ChunkConfig = field(default_factory=ChunkConfig)

    def run(self, request: RenderRequest, identity: AvatarIdentityHandle) -> RenderResult:
        chunks = list(self._chunks_for(request))
        results: List[RenderChunkResult] = []
        failed_chunks: List[int] = []
        for chunk_req in chunks:
            result = self._render_one_chunk(chunk_req, identity, request.tier)
            if result is None:
                failed_chunks.append(chunk_req.chunk_index)
                LOG.warning(
                    "Chunk %d of job %s failed after %d retries",
                    chunk_req.chunk_index,
                    request.job_id,
                    self.chunk_config.chunk_retry_max,
                )
                # Still produce a degraded result so the manifest is complete.
                result = self._degraded_chunk_result(chunk_req)
            results.append(result)

        if failed_chunks:
            LOG.warning(
                "Job %s completed with %d/%d chunks degraded",
                request.job_id,
                len(failed_chunks),
                len(chunks),
            )

        # Write a concat manifest so the EncodingWorker can assemble the
        # final video with proper overlap trimming.
        manifest_path = _write_chunk_manifest(results, request)
        return RenderResult(
            job_id=request.job_id,
            identity_id=identity.identity_id,
            output_path=manifest_path,  # manifest, not the final mp4
            duration_seconds=sum(r.duration_seconds for r in results),
            fps=request.render_spec.fps,
            tier=request.tier,
            engine_id=self.engine.engine_id,
            gpu_seconds_total=sum(r.gpu_seconds for r in results),
            completed_at=datetime.now(timezone.utc),
            chunks=tuple(results),
            degraded_chunks=tuple(failed_chunks),
        )

    # ── per-chunk render with retry ────────────────────────────────

    def _render_one_chunk(
        self,
        chunk_req: RenderChunkRequest,
        identity: AvatarIdentityHandle,
        tier: Tier,
    ) -> Optional[RenderChunkResult]:
        """Render one chunk with up to ``chunk_retry_max`` attempts.

        Returns None if all retries are exhausted.
        """
        cfg = self.chunk_config
        last_error: Optional[str] = None

        for attempt in range(cfg.chunk_retry_max):
            try:
                with self.telemetry.span(
                    "render_chunk", engine_id=self.engine.engine_id.value
                ):
                    result = self.engine.render_chunk(chunk_req, identity)

                # Publish per-chunk GPU telemetry IMMEDIATELY so costs
                # are recorded even if a later chunk fails.
                # NOTE: publish_metrics() accumulates into both the
                # in-process dataclass AND the Prometheus counter.
                # Do NOT also call record() — that would double-count.
                self.telemetry.publish_metrics(
                    engine_id=result.engine_id.value,
                    tier=tier.value,
                    gpu_seconds=result.gpu_seconds,
                    output_minutes=result.duration_seconds / 60.0,
                )
                return result

            except Exception as exc:
                last_error = str(exc)
                if attempt < cfg.chunk_retry_max - 1:
                    LOG.debug(
                        "Chunk %d attempt %d/%d failed: %s; retrying...",
                        chunk_req.chunk_index,
                        attempt + 1,
                        cfg.chunk_retry_max,
                        last_error,
                    )
                    if cfg.chunk_retry_delay_seconds > 0:
                        time.sleep(cfg.chunk_retry_delay_seconds)

        LOG.error(
            "Chunk %d exhausted all %d retries; last error: %s",
            chunk_req.chunk_index,
            cfg.chunk_retry_max,
            last_error,
        )
        return None

    def _degraded_chunk_result(
        self, chunk_req: RenderChunkRequest
    ) -> RenderChunkResult:
        """Produce a minimal result for a chunk that failed all retries."""
        duration = max(0.5, chunk_req.audio_window[1] - chunk_req.audio_window[0])
        out_dir = self.engine.settings.capture_dir / chunk_req.job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"chunk_{chunk_req.chunk_index:04d}.degraded.mp4"
        from providers._ffmpeg import FACE_REGION_RESOLUTION, write_dummy_mp4
        resolution = FACE_REGION_RESOLUTION if chunk_req.face_region_only else (512, 512)
        write_dummy_mp4(
            out_path, duration=duration, fps=chunk_req.fps, colour="0x330000", resolution=resolution
        )
        return RenderChunkResult(
            chunk_index=chunk_req.chunk_index,
            output_path=out_path,
            duration_seconds=duration,
            frames_rendered=int(duration * chunk_req.fps),
            gpu_seconds=0.0,
            engine_id=self.engine.engine_id,
        )

    # ── chunking ───────────────────────────────────────────────────

    def _chunks_for(self, request: RenderRequest) -> Iterable[RenderChunkRequest]:
        cfg = self.chunk_config
        audio_duration = _probe_audio_duration(request.render_spec.audio_path)
        if audio_duration <= 0:
            max_allowed = min(cfg.max_chunks, 16)
        else:
            effective_step = cfg.chunk_seconds - cfg.overlap_seconds
            if effective_step <= 0:
                effective_step = cfg.chunk_seconds
            expected = math.ceil(audio_duration / effective_step)
            max_allowed = min(expected + 1, cfg.max_chunks)
        index = 0
        start = 0.0
        while index < max_allowed:
            end = start + cfg.chunk_seconds
            if audio_duration > 0 and start >= audio_duration:
                break
            yield RenderChunkRequest(
                job_id=request.job_id,
                audio_window=(start, end),
                audio_path=request.render_spec.audio_path,
                fps=request.render_spec.fps,
                resolution=request.render_spec.target_resolution,
                chunk_index=index,
                overlap_seconds=cfg.overlap_seconds if start > 0 else 0.0,
                face_region_only=request.render_spec.face_region_only,
            )
            start = end - cfg.overlap_seconds if cfg.overlap_seconds > 0 else end
            index += 1


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _write_chunk_manifest(results: List[RenderChunkResult], request: RenderRequest) -> Path:
    """Write a concat manifest listing chunk paths for the EncodingWorker."""
    dest = Path("./captures") / f"{request.job_id}.manifest.txt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        fh.write(f"# manifest for job {request.job_id}\n")
        fh.write(f"# fps={request.render_spec.fps}\n")
        for r in results:
            fh.write(f"{r.chunk_index}|{r.output_path.resolve().as_posix()}|{r.duration_seconds}\n")
    return dest
