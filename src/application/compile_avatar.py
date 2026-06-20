"""Avatar-compilation use case.

This module wires together the source photo, the engine adapter, and the
pack writer. It runs once per identity and is amortised across all
subsequent videos for that identity.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from contracts.avatar_engine import AvatarEngine
from src.domain.avatar_pack import write_pack
from src.domain.types import (
    AvatarIdentityHandle,
    AvatarPackManifest,
    IdentityId,
    IdentitySpec,
)


@dataclass(slots=True, frozen=True)
class AvatarCompiler:
    """Builds an Avatar Pack from a source image via a specific engine."""

    engine: AvatarEngine
    pack_root: Path

    def compile(
        self,
        spec: IdentitySpec,
        *,
        compatibility: Optional[Tuple[str, ...]] = None,
    ) -> AvatarIdentityHandle:
        """Run the on-boarding pipeline synchronously and persist the pack."""
        spec.source_image.parent.mkdir(parents=True, exist_ok=True)
        digest = _identity_digest(spec)
        identity_id = IdentityId(f"id-{digest}")
        pack_path = self.pack_root / f"{identity_id}.tar"
        pack_path.parent.mkdir(parents=True, exist_ok=True)

        engine_id = self.engine.engine_id
        engine_id_value = engine_id.value if hasattr(engine_id, "value") else str(engine_id)

        # Produce assets. The real adapter would do face detection,
        # segmentation, keypoint extraction, VAE-encoding the latent, and
        # producing the identity embedding. Here we delegate the heavy
        # lifting to ``prepare_identity`` and let the engine write its
        # intermediates, then package them into a pack deterministically.
        intermediates = self.engine.prepare_identity(spec.source_image)

        # ``intermediates`` is a dict of {entry_name: bytes}. ``prepare_identity``
        # already ran engine-specific code; only packaging is left.
        assets: dict = dict(intermediates)
        assets["_source_image_sha256"] = hashlib.sha256(spec.source_image.read_bytes()).hexdigest()

        pack = write_pack(
            archive_path=pack_path,
            identity_id=identity_id,
            assets=assets,
            engine_compatibility=(engine_id_value, *(compatibility or ())),
        )

        return AvatarIdentityHandle(
            identity_id=identity_id,
            pack_path=pack_path,
            pack_digest=pack.digest(),
            prepared_at=pack.manifest.created_at,
            manifest_version=1,
        )


def _identity_digest(spec: IdentitySpec) -> str:
    """Deterministic hash of the source identity — used as a cache key."""
    h = hashlib.sha256()
    h.update(str(spec.source_image.resolve()).encode("utf-8"))
    h.update(b"|")
    h.update(spec.display_name.encode("utf-8"))
    h.update(b"|")
    h.update(spec.language_hint.encode("utf-8"))
    return h.hexdigest()[:16]
