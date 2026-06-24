"""Core domain types for Heyavatar.

All types here are pure data containers — no IO, no engine calls, no
loguru. They flow between the API, scheduler, workers, and adapters,
and stay dataclass-based (``slots=True``) so the GPU worker can move
thousands per second without GC churn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import NewType, Optional

from src.domain.enums import EngineId, Tier

RenderJobId = NewType("RenderJobId", str)
IdentityId = NewType("IdentityId", str)
BucketKey = NewType("BucketKey", str)


@dataclass(slots=True, frozen=True)
class IdentitySpec:
    """Specification for the source identity (photo + optional metadata)."""

    source_image: Path
    display_name: str = ""
    language_hint: str = ""
    safe_motion_overrides: Optional[dict] = None


@dataclass(slots=True, frozen=True)
class RenderSpec:
    """What the user actually wants to render."""

    audio_path: Path
    fps: int = 25
    target_resolution: tuple[int, int] = (512, 512)
    background_path: Optional[Path] = None
    face_region_only: bool = False  # output face crop at native 256×256, skip upscale/pasteback


@dataclass(slots=True, frozen=True)
class RenderRequest:
    """Top-level request issued from the API."""

    job_id: RenderJobId
    identity_id: IdentityId
    identity_spec: IdentitySpec
    render_spec: RenderSpec
    tier: Tier = Tier.EXPRESS
    requested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    callback_url: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class AvatarIdentityHandle:
    """Materialised handle returned by :meth:`AvatarEngine.prepare_identity`."""

    identity_id: IdentityId
    pack_path: Path
    pack_digest: str
    prepared_at: datetime
    manifest_version: int = 1


@dataclass(slots=True, frozen=True)
class RenderChunkRequest:
    """Per-chunk render request inside a long-form video."""

    job_id: RenderJobId
    audio_window: tuple[float, float]  # (start_seconds, end_seconds)
    audio_path: Path
    fps: int = 25
    resolution: tuple[int, int] = (512, 512)
    chunk_index: int = 0
    overlap_seconds: float = 0.0
    face_region_only: bool = False  # render face crop only, save VRAM
    face_motion_timeline_path: Optional[Path] = None


@dataclass(slots=True, frozen=True)
class RenderChunkResult:
    """Per-chunk output."""

    chunk_index: int
    output_path: Path
    duration_seconds: float
    frames_rendered: int
    gpu_seconds: float
    engine_id: EngineId


@dataclass(slots=True, frozen=True)
class RenderResult:
    """Final aggregated response once a video is finished."""

    job_id: RenderJobId
    identity_id: IdentityId
    output_path: Path
    duration_seconds: float
    fps: int
    tier: Tier
    engine_id: EngineId
    gpu_seconds_total: float
    completed_at: datetime
    chunks: tuple[RenderChunkResult, ...]
    degraded_chunks: tuple[int, ...] = ()


@dataclass(slots=True, frozen=True)
class AvatarPackManifest:
    """Self-describing metadata stored alongside an Avatar Pack."""

    identity_id: IdentityId
    source_image_sha256: str
    created_at: datetime
    entry_files: tuple[str, ...]
    engine_compatibility: tuple[EngineId, ...]
    safe_motion_ranges: dict = field(default_factory=dict)
    color_profile_sha256: str = ""
    notes: str = ""

    def iter_entries(self) -> tuple[str, ...]:
        return self.entry_files
