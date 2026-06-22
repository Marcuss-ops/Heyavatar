"""Pydantic schemas for jobs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from contracts.job_queue import JobState, RenderJob
from src.domain.types import RenderJobId

# The three job flavours the GPU worker (``workers/gpu_worker/process.py``)
# dispatches on. ``compile`` is registered as a job type so existing callers
# that POST ``job_type=compile`` keep working; ``render_cached`` is the
# opt-in path the worker added in Change 4 / §6 of the slimming plan.
JOB_TYPE_RENDER = "render"
JOB_TYPE_COMPILE = "compile"
JOB_TYPE_RENDER_CACHED = "render_cached"


class JobSubmitRequest(BaseModel):
    identity_id: str = Field(..., description="Identity produced by /avatars/compile.")
    # ``source_image`` is REQUIRED for the legacy ``render`` and ``compile``
    # paths (it's how the renderer + compiler compile work), but OPTIONAL
    # for ``render_cached`` — that path can use an existing identity pack
    # and only needs the body-template keys. The validator below enforces
    # the conditional requirement. Default ``None`` keeps it permissive.
    source_image: Optional[str] = Field(
        None,
        description=(
            "Path or URL to the source face photo. "
            "Required for job_type 'render' / 'compile'; "
            "optional for job_type 'render_cached' (identity pack may already exist)."
        ),
    )
    audio_path: str = Field(..., description="Path to the audio file to drive the video.")
    fps: int = Field(25, ge=1, le=60)
    tier: str = Field("express", description="One of: express | studio | premium.")
    callback_url: Optional[str] = Field(None, description="Optional webhook for completion.")
    client_reference: Optional[str] = Field(None, description="User-supplied id for tracing.")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # ── Cached-render opt-in fields ─────────────────────────────
    # ``job_type`` selects the worker dispatcher path. The default keeps
    # the legacy chunked ``RenderVideo`` flow; setting ``render_cached``
    # routes through ``src.application.render_cached_avatar`` and reads
    # ``avatar_id`` / ``gesture_id`` from the body template tree.
    job_type: Literal["render", "compile", "render_cached"] = Field(
        "render",
        description=(
            "Worker dispatcher path. "
            "'render' = legacy chunked RenderVideo; "
            "'compile' = identity pack registration; "
            "'render_cached' = body template + face-region MuseTalk + compositor + QC."
        ),
    )
    avatar_id: Optional[str] = Field(
        None,
        description="Required when job_type=render_cached; key into body_templates/<avatar_id>/.",
    )
    gesture_id: Optional[str] = Field(
        None,
        description="Required when job_type=render_cached; key into body_templates/<avatar_id>/<gesture_id>/.",
    )

    @model_validator(mode="after")
    def _validate_cached_job(self) -> "JobSubmitRequest":
        """Reject missing keys synchronously at the API layer.

        Rules:

        * ``job_type=render_cached``  → ``avatar_id`` + ``gesture_id`` required;
          ``source_image`` is OPTIONAL (the use case can run with an
          existing identity pack).
        * ``job_type in ('render', 'compile')`` → ``source_image`` required
          (legacy behaviour; matches what the render path already assumes).
        * ``audio_path`` is ALWAYS required regardless of ``job_type``.

        The worker also re-checks (see ``GpuWorker._do_process_render_cached``)
        so a payload published directly to the queue rejects cleanly too.
        Returning ``self`` keeps the model immutable; raising
        :class:`ValueError` surfaces as a 422 to the FastAPI client.
        """
        if self.job_type == JOB_TYPE_RENDER_CACHED:
            missing = [
                name for name, value in (
                    ("avatar_id", self.avatar_id),
                    ("gesture_id", self.gesture_id),
                ) if not value
            ]
            if missing:
                raise ValueError(
                    f"job_type=render_cached requires these fields: "
                    f"{', '.join(missing)}"
                )
        else:
            # Legacy ``render`` + ``compile`` paths still need a source image.
            if not self.source_image:
                raise ValueError(
                    f"job_type={self.job_type!r} requires source_image."
                )
        return self

    def to_queue_payload(self) -> Dict[str, Any]:
        return {
            "job_type": self.job_type,
            "identity_id": self.identity_id,
            "source_image": self.source_image,
            "audio_path": self.audio_path,
            "fps": self.fps,
            "tier": self.tier,
            "callback_url": self.callback_url,
            "client_reference": self.client_reference,
            "metadata": self.metadata,
            "avatar_id": self.avatar_id,
            "gesture_id": self.gesture_id,
        }


class JobSubmitResponse(BaseModel):
    job_id: RenderJobId
    state: JobState


class JobResponse(BaseModel):
    job_id: RenderJobId
    state: JobState
    attempts: int
    reserved_by: Optional[str]
    last_error: Optional[str]
    created_at: datetime
    updated_at: datetime
    # ── result fields (populated when job is terminal) ─────────
    identity_id: Optional[str] = None
    output_path: Optional[str] = None
    result_url: Optional[str] = None
    duration_seconds: Optional[float] = None
    engine_id: Optional[str] = None
    gpu_seconds: Optional[float] = None
    degraded: Optional[bool] = None
    # ── render_cached result fields (only populated for that job type) ─
    avatar_id: Optional[str] = None
    gesture_id: Optional[str] = None
    qc_status: Optional[str] = None
    batch_size: Optional[int] = None
    face_resolution: Optional[tuple[int, int]] = None
    wall_seconds: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None

    @classmethod
    def from_job(cls, job: RenderJob) -> "JobResponse":
        result = job.result or {}
        output_path = result.get("output_path")
        # Derive result_url from the actual output_path stored by the worker.
        if output_path:
            result_url = f"/captures/{Path(output_path).name}"
        else:
            result_url = None
        face_res = result.get("face_resolution")
        return cls(
            job_id=job.id,
            state=job.state,
            attempts=job.attempts,
            reserved_by=job.reserved_by,
            last_error=job.last_error,
            created_at=job.created_at,
            updated_at=job.updated_at,
            identity_id=result.get("identity_id"),
            output_path=output_path,
            result_url=result_url,
            duration_seconds=result.get("duration_seconds"),
            engine_id=result.get("engine_id"),
            gpu_seconds=result.get("gpu_seconds"),
            degraded=result.get("degraded"),
            avatar_id=result.get("avatar_id"),
            gesture_id=result.get("gesture_id"),
            qc_status=result.get("qc_status"),
            batch_size=result.get("batch_size"),
            face_resolution=tuple(face_res) if face_res else None,
            wall_seconds=result.get("wall_seconds"),
            metrics=result.get("metrics"),
        )
