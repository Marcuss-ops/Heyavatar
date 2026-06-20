"""Pydantic schemas for jobs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from contracts.job_queue import JobState, RenderJob
from src.domain.types import RenderJobId


class JobSubmitRequest(BaseModel):
    identity_id: str = Field(..., description="Identity produced by /avatars/compile.")
    source_image: str = Field(..., description="Path or URL to the source face photo.")
    audio_path: str = Field(..., description="Path to the audio file to drive the video.")
    fps: int = Field(25, ge=1, le=60)
    tier: str = Field("express", description="One of: express | studio | premium.")
    callback_url: Optional[str] = Field(None, description="Optional webhook for completion.")
    client_reference: Optional[str] = Field(None, description="User-supplied id for tracing.")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_queue_payload(self) -> Dict[str, Any]:
        return {
            "identity_id": self.identity_id,
            "source_image": self.source_image,
            "audio_path": self.audio_path,
            "fps": self.fps,
            "tier": self.tier,
            "callback_url": self.callback_url,
            "client_reference": self.client_reference,
            "metadata": self.metadata,
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

    @classmethod
    def from_job(cls, job: RenderJob) -> "JobResponse":
        result = job.result or {}
        output_path = result.get("output_path")
        # Derive result_url from the actual output_path stored by the worker.
        if output_path:
            result_url = f"/captures/{Path(output_path).name}"
        else:
            result_url = None
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
        )
