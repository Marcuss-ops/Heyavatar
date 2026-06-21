"""Avatar endpoints: compile (post) and list (get).

Avatar compilation is dispatched as a queue job so the GPU worker
(not the FastAPI gateway) loads the engine. The ``POST /avatars/compile``
returns a ``job_id`` that the caller polls via ``GET /jobs/{job_id}``.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request, status

from api.schemas.avatars import AvatarCompileRequest, AvatarCompileResponse, AvatarSummary
from contracts.job_queue import JobState, RenderJob
from src.domain.enums import EngineId, Tier
from src.domain.avatar_pack import read_pack
from src.domain.types import RenderJobId

router = APIRouter(prefix="/avatars", tags=["avatars"])


@router.post("/compile", response_model=AvatarCompileResponse,
             status_code=status.HTTP_202_ACCEPTED)
def compile_avatar(payload: AvatarCompileRequest, request: Request) -> AvatarCompileResponse:
    state = request.app.state.deps
    engine_id = EngineId.from_string(payload.engine_id) if payload.engine_id else None
    if engine_id is None:
        engine_id = EngineId.MUSE_TALK
    source = Path(payload.source_image)
    if not source.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"source_image not found at {source}",
        )
    job_id = RenderJobId(f"job-compile-{uuid.uuid4().hex[:16]}")
    queue_payload = {
        "job_type": "compile",
        "engine_id": engine_id.value,
        "source_image": str(source.resolve()),
        "display_name": payload.display_name or "",
        "language_hint": payload.language_hint or "",
    }
    try:
        from src.observability.distributed.propagation import inject_traceparent
        queue_payload = inject_traceparent(queue_payload)
    except ImportError:
        pass
    try:
        from src.observability.distributed.tracing import get_tracer
        tracer = get_tracer("api.avatars")
        with tracer.start_as_current_span("api.compile_avatar", attributes={
            "heyavatar.job_id": str(job_id),
            "heyavatar.engine_id": engine_id.value,
        }):
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
    return AvatarCompileResponse(
        job_id=job_id,
        engine_id=engine_id,
        state=JobState.PENDING,
    )


@router.get("/{identity_id}", response_model=AvatarSummary)
def get_avatar(identity_id: str, request: Request) -> AvatarSummary:
    state = request.app.state.deps
    pack = state.pack_repo.get(identity_id)  # type: ignore[arg-type]
    if pack is None:
        raise HTTPException(status_code=404, detail=f"identity '{identity_id}' not found")
    return AvatarSummary(
        identity_id=pack.manifest.identity_id,
        engine_compatibility=list(pack.manifest.engine_compatibility),
        entry_files=list(pack.manifest.entry_files),
        created_at=pack.manifest.created_at,
        notes=pack.manifest.notes,
    )


@router.get("", response_model=List[AvatarSummary])
def list_avatars(request: Request) -> List[AvatarSummary]:
    state = request.app.state.deps
    out: List[AvatarSummary] = []
    for path in sorted(state.pack_repo.root.glob("*.tar")):
        try:
            pack = read_pack(path)
        except (ValueError, OSError):
            continue
        out.append(AvatarSummary(
            identity_id=pack.manifest.identity_id,
            engine_compatibility=list(pack.manifest.engine_compatibility),
            entry_files=list(pack.manifest.entry_files),
            created_at=pack.manifest.created_at,
            notes=pack.manifest.notes,
        ))
    return out
