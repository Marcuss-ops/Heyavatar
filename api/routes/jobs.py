"""Job endpoints: submit a render, check status, cancel.

Each route is wrapped in a server-kind OpenTelemetry span and the
W3C ``traceparent`` is injected into the published job payload so the
worker process can resume the same trace.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Response, status

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


def _inject_traceparent(payload: dict) -> dict:
    try:
        from src.observability.distributed.propagation import inject_traceparent
        return inject_traceparent(payload)
    except ImportError:
        return payload


@router.post("", response_model=JobSubmitResponse, status_code=status.HTTP_202_ACCEPTED)
def submit_job(payload: JobSubmitRequest, request: Request) -> JobSubmitResponse:
    state = request.app.state.deps
    job_id = _new_job_id(payload.client_reference)
    queue_payload = payload.to_queue_payload()
    tier_value = queue_payload.get("tier", "express")
    # W3C traceparent gets stamped onto the payload so the worker can
    # continue the trace it never sees natively.
    queue_payload = _inject_traceparent(queue_payload)

    # Server-kind span so the producer side of the trace is linkable.
    try:
        from src.observability.distributed.tracing import get_tracer
        tracer = get_tracer("api.jobs")
        with tracer.start_as_current_span(
            "api.submit_job",
            attributes={
                "heyavatar.job_id": str(job_id),
                "heyavatar.tier": str(tier_value),
                "heyavatar.motion_style": str(queue_payload.get("motion_style") or "natural"),
            },
        ):
            job = RenderJob(
                id=job_id,
                state=JobState.PENDING,
                payload=queue_payload,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            state.queue.publish(job)
            state.job_repo.upsert(job)
    except ImportError:
        job = RenderJob(
            id=job_id,
            state=JobState.PENDING,
            payload=queue_payload,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        state.queue.publish(job)
        state.job_repo.upsert(job)

    try:
        from src.observability.metrics.recorders import record_terminal
        record_terminal(state="pending", tier=str(tier_value))
    except ImportError:
        pass

    return JobSubmitResponse(job_id=job_id, state=JobState.PENDING)


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str, request: Request) -> JobResponse:
    state = request.app.state.deps
    job = state.job_repo.get(RenderJobId(job_id))
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JobResponse.from_job(job)


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def cancel_job(job_id: str, request: Request) -> Response:
    state = request.app.state.deps
    job_id_t = RenderJobId(job_id)
    state.queue.cancel(job_id_t)
    state.job_repo.mark(job_id_t, JobState.CANCELLED)
    try:
        from src.observability.metrics.recorders import record_terminal
        record_terminal(state="cancelled", tier="express")
    except ImportError:
        pass
    return Response(status_code=status.HTTP_204_NO_CONTENT)
