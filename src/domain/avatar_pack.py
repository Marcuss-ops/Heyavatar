"""Avatar Pack: per-identity pre-computed bundle.

The real competitive advantage of the platform: instead of doing face
detection, segmentation, VAE encoding, identity analysis on every video,
we bake all that work into a per-identity **Avatar Pack** and reuse it.

Stored as an ``.tar`` archive on disk (LZ4-compressed for fast IO) with
a structured ``manifest.json`` at the root. The pack contains the source
features, canonical keypoints, face mask, face crop, identity embedding,
VAE-encoded source latent, optional background, colour profile, and
safe motion ranges.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from src.domain.types import AvatarPackManifest, IdentityId


PACK_VERSION = 1
# Default required entries (LivePortrait shape). Callers can override
# via ``required_entries`` when an engine stores a different bundle.
PACK_ENTRY_REQUIRED_DEFAULT = (
    "manifest.json",
    "source_features.bin",
    "canonical_keypoints.bin",
    "face_mask.png",
    "face_crop.png",
    "identity_embedding.bin",
    "source_latent.bin",
)
PACK_ENTRY_OPTIONAL = (
    "background.png",
    "color_profile.json",
    "safe_motion_ranges.json",
)


@dataclass(slots=True, frozen=True)
class AvatarPack:
    """In-memory representation of a pack directory or tar archive."""

    manifest: AvatarPackManifest
    archive_path: Path

    def digest(self) -> str:
        """SHA-256 of the archive bytes — used as a content-addressed key."""
        h = hashlib.sha256()
        with open(self.archive_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()


def write_pack(
    *,
    archive_path: Path,
    identity_id: IdentityId,
    assets: dict,
    safe_motion_ranges: Optional[dict] = None,
    engine_compatibility: Iterable[str] = (),
    required_entries: Iterable[str] = PACK_ENTRY_REQUIRED_DEFAULT,
) -> AvatarPack:
    """Create a new pack from a dictionary of {entry_name: bytes/str | Path} and return it.

    *required_entries* can be overridden per engine; MuseTalk passes its
    own bundle (“source_latent.bin” instead of “source_features.bin”).
    """
    # Validate required entries *before* touching the filesystem so callers
    # receive one clear error rather than a half-written archive.
    required = tuple(required_entries)
    provided = {name for name in assets if not name.startswith("_")}
    missing = [name for name in required
               if name != "manifest.json" and name not in provided]
    if missing:
        raise KeyError(
            "Avatar pack is missing required entries: "
            + ", ".join(missing)
        )

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = AvatarPackManifest(
        identity_id=identity_id,
        source_image_sha256=assets.get("_source_image_sha256", ""),
        created_at=datetime.now(timezone.utc),
        entry_files=tuple(sorted(assets.keys())),
        engine_compatibility=tuple(engine_compatibility),
        safe_motion_ranges=dict(safe_motion_ranges or {}),
        notes=assets.get("_notes", ""),
    )

    manifest_json = json.dumps(_manifest_to_dict(manifest), indent=2).encode("utf-8")

    tmp = archive_path.with_suffix(archive_path.suffix + f".{os.getpid()}.tmp")
    with tarfile.open(tmp, mode="w") as tf:
        _add_bytes(tf, "manifest.json", manifest_json)

        for entry, value in assets.items():
            if entry.startswith("_"):
                continue
            if isinstance(value, (bytes, bytearray)):
                _add_bytes(tf, entry, bytes(value))
            elif isinstance(value, str):
                _add_bytes(tf, entry, value.encode("utf-8"))
            elif isinstance(value, Path):
                if value.is_file():
                    tf.add(value, arcname=entry, recursive=False)
                else:
                    raise FileNotFoundError(f"Pack asset {entry} not found at {value}")
            else:
                raise TypeError(f"Unsupported asset type for entry '{entry}': {type(value).__name__}")

    os.replace(tmp, archive_path)
    return AvatarPack(manifest=manifest, archive_path=archive_path)


def read_pack(archive_path: Path) -> AvatarPack:
    """Read the manifest out of an archive and return an :class:`AvatarPack`."""
    if not archive_path.is_file():
        raise FileNotFoundError(f"Avatar pack archive not found at {archive_path}")
    with tarfile.open(archive_path, mode="r") as tf:
        try:
            manifest_member = tf.getmember("manifest.json")
        except KeyError as exc:
            raise ValueError(f"Pack at {archive_path} is missing manifest.json") from exc
        manifest_io = tf.extractfile(manifest_member)
        if manifest_io is None:
            raise ValueError(f"Pack at {archive_path} has unreadable manifest.json")
        manifest_dict = json.loads(manifest_io.read().decode("utf-8"))
    manifest = _dict_to_manifest(manifest_dict)
    return AvatarPack(manifest=manifest, archive_path=archive_path)


def read_pack_asset(archive_path: Path, entry: str) -> bytes:
    """Return the bytes for a single pack entry. Raises if missing."""
    with tarfile.open(archive_path, mode="r") as tf:
        try:
            member = tf.getmember(entry)
        except KeyError as exc:
            raise KeyError(f"Pack entry '{entry}' missing in {archive_path}") from exc
        data = tf.extractfile(member)
        if data is None:
            raise ValueError(f"Pack entry '{entry}' is not a regular file in {archive_path}")
        return data.read()


def verify_pack(archive_path: Path) -> List[str]:
    """Return a list of missing-required entries; empty means the pack is valid."""
    try:
        pack = read_pack(archive_path)
    except (FileNotFoundError, ValueError, tarfile.TarError) as exc:
        return [f"unreadable: {exc}"]
    missing = [name for name in PACK_ENTRY_REQUIRED_DEFAULT if name not in pack.manifest.entry_files
               and name != "manifest.json"]
    with tarfile.open(archive_path, mode="r") as tf:
        names = set(tf.getnames())
    return [name for name in missing if name not in names]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_bytes(tf: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mtime = int(time.time())
    info.mode = 0o644
    tf.addfile(info, io.BytesIO(data))


def _manifest_to_dict(manifest: AvatarPackManifest) -> dict:
    return {
        "version": PACK_VERSION,
        "identity_id": manifest.identity_id,
        "source_image_sha256": manifest.source_image_sha256,
        "created_at": manifest.created_at.isoformat(),
        "entry_files": list(manifest.entry_files),
        "engine_compatibility": list(manifest.engine_compatibility),
        "safe_motion_ranges": dict(manifest.safe_motion_ranges),
        "color_profile_sha256": manifest.color_profile_sha256,
        "notes": manifest.notes,
    }


def _dict_to_manifest(d: dict) -> AvatarPackManifest:
    return AvatarPackManifest(
        identity_id=IdentityId(d["identity_id"]),
        source_image_sha256=d.get("source_image_sha256", ""),
        created_at=datetime.fromisoformat(d["created_at"]),
        entry_files=tuple(d.get("entry_files", ())),
        engine_compatibility=tuple(d.get("engine_compatibility", ())),
        safe_motion_ranges=dict(d.get("safe_motion_ranges", {})),
        color_profile_sha256=d.get("color_profile_sha256", ""),
        notes=d.get("notes", ""),
    )
