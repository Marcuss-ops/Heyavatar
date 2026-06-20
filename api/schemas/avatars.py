"""Pydantic schemas for avatars."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from src.domain.enums import EngineId
from src.domain.types import IdentityId


class AvatarCompileRequest(BaseModel):
    source_image: str = Field(..., description="Local path or URL to the source photo.")
    display_name: str = Field("", description="Optional human-friendly label.")
    language_hint: str = Field("", description="Optional BCP-47 language hint, e.g. 'it-IT'.")
    engine_id: Optional[str] = Field(
        None, description="Override the default engine (must exist in registry)."
    )


class AvatarCompileResponse(BaseModel):
    identity_id: IdentityId
    engine_id: EngineId
    pack_path: str
    pack_digest: str
    prepared_at: datetime


class AvatarSummary(BaseModel):
    identity_id: IdentityId
    engine_compatibility: List[str]
    entry_files: List[str]
    created_at: datetime
    notes: str
