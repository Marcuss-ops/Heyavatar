"""Concrete body template data model and filesystem loader.

The cached avatar pipeline only needs a prerecorded body clip plus its
mask and transform assets. Keep the model intentionally small and
filesystem-oriented so runtime code stays deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BodyTemplate:
    body_video: Path
    face_mask: Path
    neck_mask: Path
    face_transforms: Path
    metadata: Path


def load_body_template(
    avatar_id: str,
    gesture_id: str,
    *,
    base_dir: str | Path = "avatar_packs",
) -> BodyTemplate:
    """Load a cached body template from disk.

    Resolution order:
    1. ``<base_dir>/<avatar_id>/body_cache/<gesture_id>/``
    2. fallback ``avatar_packs/<avatar_id>/body_cache/<gesture_id>/``
    """
    root = Path(base_dir)
    candidate_dirs = [
        root / avatar_id / "body_cache" / gesture_id,
        Path("avatar_packs") / avatar_id / "body_cache" / gesture_id,
    ]
    for pack_dir in candidate_dirs:
        body_video = pack_dir / "body.mp4"
        face_mask = pack_dir / "face_mask.mp4"
        neck_mask = pack_dir / "neck_mask.mp4"
        face_transforms = pack_dir / "face_transforms.npz"
        metadata = pack_dir / "metadata.json"
        if body_video.is_file() and face_mask.is_file() and neck_mask.is_file() and face_transforms.is_file():
            return BodyTemplate(
                body_video=body_video,
                face_mask=face_mask,
                neck_mask=neck_mask,
                face_transforms=face_transforms,
                metadata=metadata,
            )
    raise FileNotFoundError(
        f"Body template not found for avatar_id={avatar_id!r}, gesture_id={gesture_id!r}"
    )
