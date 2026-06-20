"""Pydantic schemas for jobs."""

from __future__ import annotations

from datetime import datetime
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

    @classmethod
    def from_job(cls, job: RenderJob) -> "JobResponse":
        return cls(
            job_id=job.id,
            state=job.state,
            attempts=job.attempts,
            reserved_by=job.reserved_by,
            last_error=job.last_error,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )
