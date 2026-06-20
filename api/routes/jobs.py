"""Job endpoints: submit a render, check status, cancel."""

from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, status

from api.schemas.jobs import (
    JobResponse,
    JobSubmitRequest,
    JobSubmitResponse,
)
from contracts.job_queue import JobState, RenderJob
from src.domain.types import (
    IdentitySpec,
    RenderJobId,
    RenderRequest,
    RenderSpec,
)
from src.domain.enums import Tier

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _new_job_id(client_reference: Optional[str]) -> RenderJobId:
    """Build a job id. Prefer ``client_reference``; otherwise uuid4().hex."""
    if client_reference:
        return RenderJobId(f"job-{client_reference}")
    return RenderJobId(f"job-{uuid.uuid4().hex}")


@router.post("", response_model=JobSubmitResponse, status_code=status.HTTP_202_ACCEPTED)
def submit_job(payload: JobSubmitRequest, request: Request) -> JobSubmitResponse:
    state = request.app.state.deps
    job_id = _new_job_id(payload.client_reference)
    job = RenderJob(
        id=job_id,
        state=JobState.PENDING,
        payload=payload.to_queue_payload(),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    state.queue.publish(job)
    state.job_repo.upsert(job)
    return JobSubmitResponse(job_id=job_id, state=JobState.PENDING)


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str, request: Request) -> JobResponse:
    state = request.app.state.deps
    job = state.job_repo.get(RenderJobId(job_id))
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JobResponse.from_job(job)


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def cancel_job(job_id: str, request: Request) -> None:
    state = request.app.state.deps
    job_id_t = RenderJobId(job_id)
    state.queue.cancel(job_id_t)
    state.job_repo.mark(job_id_t, JobState.CANCELLED)
