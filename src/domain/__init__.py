"""Pure data types for the avatar engine."""

from .avatar_pack import AvatarPack, read_pack, read_pack_asset, verify_pack, write_pack
from .body_template import BodyTemplate, load_body_template
from .enums import EngineId, Tier
from .timeline import DEFAULT_TIMELINE_FPS, Timeline, TimelineSegment
from .types import (
    AvatarIdentityHandle,
    AvatarPackManifest,
    BucketKey,
    IdentityId,
    IdentitySpec,
    RenderChunkRequest,
    RenderChunkResult,
    RenderJobId,
    RenderRequest,
    RenderResult,
    RenderSpec,
)

__all__ = [
    "AvatarIdentityHandle",
    "AvatarPack",
    "AvatarPackManifest",
    "BodyTemplate",
    "BucketKey",
    "DEFAULT_TIMELINE_FPS",
    "EngineId",
    "IdentityId",
    "IdentitySpec",
    "RenderChunkRequest",
    "RenderChunkResult",
    "RenderJobId",
    "RenderRequest",
    "RenderResult",
    "RenderSpec",
    "Tier",
    "Timeline",
    "TimelineSegment",
    "load_body_template",
    "read_pack",
    "read_pack_asset",
    "verify_pack",
    "write_pack",
]
