"""Avatar endpoints: compile (post) and list (get)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request, status

from api.schemas.avatars import AvatarCompileRequest, AvatarCompileResponse, AvatarSummary
from contracts.avatar_engine import AvatarEngine
from providers import get_provider
from src.application.compile_avatar import AvatarCompiler
from src.domain.enums import EngineId, Tier
from src.domain.avatar_pack import read_pack
from src.domain.types import IdentitySpec

router = APIRouter(prefix="/avatars", tags=["avatars"])


@router.post("/compile", response_model=AvatarCompileResponse,
             status_code=status.HTTP_201_CREATED)
def compile_avatar(payload: AvatarCompileRequest, request: Request) -> AvatarCompileResponse:
    state = request.app.state.deps
    engine_id = EngineId.from_string(payload.engine_id) if payload.engine_id else None
    if engine_id is None:
        # Default to the registry's tier express engine
        engine_id = EngineId.MUSE_TALK
    source = Path(payload.source_image)
    if not source.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"source_image not found at {source}",
        )
    engine = get_provider(engine_id)
    engine.load()
    try:
        compiler = AvatarCompiler(engine=engine, pack_root=state.pack_repo.root)
        spec = IdentitySpec(
            source_image=source,
            display_name=payload.display_name,
            language_hint=payload.language_hint,
        )
        handle = compiler.compile(spec)
    finally:
        engine.unload()
    state.pack_repo.save(handle.identity_id, read_pack(handle.pack_path))
    return AvatarCompileResponse(
        identity_id=handle.identity_id,
        engine_id=engine_id,
        pack_path=str(handle.pack_path),
        pack_digest=handle.pack_digest,
        prepared_at=handle.prepared_at,
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
