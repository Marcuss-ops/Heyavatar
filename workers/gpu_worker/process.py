"""Per-job processing methods, attached to :class:`GpuWorker`.

This module is imported at module-end by :mod:`workers.gpu_worker.worker`
so the class returned from ``from workers.gpu_worker.worker import
GpuWorker`` is fully armed — ``_process`` and ``_do_process`` are bound
methods on the class.

Why a separate module
---------------------
``_do_process`` alone is ~140 lines and reads differently from the
outer reservation loop in :mod:`worker`. Keeping it topic-isolated
makes both files easier to diff during reviews and pipe-cleaning.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from contracts.job_queue import JobState, RenderJob
from src.application import AvatarCompiler, RenderVideo  # noqa: F401  -- used indirectly
from src.domain.types import (
    IdentityId,
    IdentitySpec,
    RenderRequest,
    RenderSpec,
)
from src.storage.avatar_packs import AvatarPackRepository
from workers.gpu_worker.telemetry import _id_from_str, read_pack_from_archive


def _process_impl(self, job: RenderJob, engine, repo: AvatarPackRepository):
    """Process a job and return ``(final_state, result_dict)``.

    Wraps ``_do_process`` in an OpenTelemetry consumer-kind span if
    OTel is installed, otherwise falls through.
    """
    tracer = None
    try:
        from src.observability.distributed.propagation import extract_traceparent
        from src.observability.distributed.tracing import get_tracer
        tracer = get_tracer("workers.gpu_worker")
        parent_ctx = extract_traceparent(job.payload)
    except ImportError:
        parent_ctx = None
        tracer = None

    if tracer is not None:
        from opentelemetry import context as otel_context, trace
        token = otel_context.attach(parent_ctx) if parent_ctx is not None else None
        try:
            with tracer.start_as_current_span(
                "worker.render_job",
                kind=trace.SpanKind.CONSUMER,
                attributes={
                    "heyavatar.engine_id": self.engine_id.value,
                    "heyavatar.worker_id": self.handle.worker_id,
                    "heyavatar.job_id": str(job.id),
                    "heyavatar.tier": str(job.payload.get("tier", "express")),
                },
            ):
                return self._do_process(job, engine, repo)
        finally:
            if token is not None:
                otel_context.detach(token)
    else:
        return self._do_process(job, engine, repo)


def _do_process_impl(self, job: RenderJob, engine, repo: AvatarPackRepository):
    """Process a job and return ``(final_state, result_dict)``.

    Handles three job flavours:

    * ``compile`` — prepare an identity from a source image and store
      it in the pack repository.
    * ``render``  — pull the identity, render audio chunks, hand the
      manifest to the encoder, and return the final mp4 path.
    """
    payload = job.payload
    job_type = payload.get("job_type", "render")

    # ── compile-only job ──────────────────────────────────────
    if job_type == "compile":
        source_image = Path(payload["source_image"])
        engine_id_value = payload.get("engine_id", self.engine_id.value)
        spec = IdentitySpec(
            source_image=source_image,
            display_name=payload.get("display_name", ""),
            language_hint=payload.get("language_hint", ""),
        )
        compiler = AvatarCompiler(engine=engine, pack_root=repo.root)
        handle = compiler.compile(spec)
        repo.save(handle.identity_id, read_pack_from_archive(handle.pack_path))
        # Emit a telemetry signal so the operator knows something happened.
        try:
            from src.observability.metrics.recorders import record_terminal
            record_terminal(state="compiled", tier="express")
        except ImportError:
            pass
        return JobState.COMPLETED, {
            "identity_id": str(handle.identity_id),
            "engine_id": engine_id_value,
            "pack_digest": handle.pack_digest,
        }

    # ── render job ────────────────────────────────────────────
    identity_id_str = payload["identity_id"]
    handle = repo.get(_id_from_str(identity_id_str))
    if handle is None:
        spec = IdentitySpec(
            source_image=Path(payload["source_image"]),
            display_name=payload.get("display_name", ""),
        )
        compiler = AvatarCompiler(engine=engine, pack_root=repo.root)
        compiled = compiler.compile(spec)
        repo.save(compiled.identity_id, read_pack_from_archive(compiled.pack_path))
        handle = repo.get(compiled.identity_id)
        if handle is None:
            raise RuntimeError(f"Failed to compile and persist identity for {identity_id_str}")
    request = RenderRequest(
        job_id=job.id,
        identity_id=IdentityId(identity_id_str),
        identity_spec=IdentitySpec(source_image=Path(payload["source_image"])),
        render_spec=RenderSpec(
            audio_path=Path(payload["audio_path"]),
            fps=int(payload.get("fps", 25)),
        ),
        tier=payload.get("tier", "express"),
    )
    rv = RenderVideo(engine=engine, telemetry=self.telemetry)
    result = rv.run(request, handle)
    # Per-chunk telemetry is published inside RenderVideo.run() —
    # no need to double-publish here. The headline metric
    # ``gpu_seconds_per_minute_of_output`` is computed in Prometheus.

    # ── determine render state ─────────────────────────────
    total_chunks = len(result.chunks)
    degraded_count = len(result.degraded_chunks)
    if degraded_count == total_chunks:
        render_state = JobState.FAILED_INFERENCE
        degraded = True
    elif degraded_count > 0:
        render_state = JobState.COMPLETED_DEGRADED
        degraded = True
    else:
        render_state = JobState.COMPLETED
        degraded = False

    # ── encoding pass ────────────────────────────────────────
    final_path: Optional[Path] = None
    manifest_path = result.output_path
    if manifest_path.is_file():
        from src.core.logging import get_logger
        from workers.encoding_worker.worker import EncodingWorker  # type: ignore
        encoder = EncodingWorker(settings=self.settings)
        try:
            final_path = encoder.encode(
                str(job.id),
                manifest_path,
                audio_path=request.render_spec.audio_path,
            )
            get_logger(__name__).info("encoded final video", extra={"path": str(final_path)})
        except Exception as exc:
            get_logger(__name__).error(
                "encoding failed", extra={"error": str(exc)}
            )
            return JobState.FAILED_ENCODING, {
                "identity_id": identity_id_str,
                "engine_id": self.engine_id.value,
                "duration_seconds": result.duration_seconds,
                "gpu_seconds": result.gpu_seconds_total,
                "degraded": degraded,
                "degraded_chunks": list(result.degraded_chunks),
                "total_chunks": total_chunks,
                "error": f"Encoding failed: {exc}",
            }

    return render_state, {
        "identity_id": identity_id_str,
        "output_path": str(final_path) if final_path else None,
        "engine_id": self.engine_id.value,
        "duration_seconds": result.duration_seconds,
        "gpu_seconds": result.gpu_seconds_total,
        "degraded": degraded,
        "degraded_chunks": list(result.degraded_chunks),
        "total_chunks": total_chunks,
    }


# ── bind to GpuWorker ──────────────────────────────────────────────
# Importing the class first then assigning the functions as attributes
# turns them into bound methods when accessed via an instance. We
# import here (rather than at module top) so the ``process`` module
# loads AFTER ``GpuWorker`` is defined.
from workers.gpu_worker.worker import GpuWorker  # noqa: E402

GpuWorker._process = _process_impl
GpuWorker._do_process = _do_process_impl
