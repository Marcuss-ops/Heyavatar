"""Avatar Pack storage — file-based repository."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.domain.avatar_pack import read_pack, write_pack, AvatarPack, verify_pack
from src.domain.types import AvatarIdentityHandle, IdentityId


@dataclass(slots=True)
class AvatarPackRepository:
    """Disk-backed repository of compiled Avatar Packs."""

    root: Path
    _cache: Dict[IdentityId, AvatarPack] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, identity_id: IdentityId) -> Path:
        return self.root / f"{identity_id}.tar"

    def exists(self, identity_id: IdentityId) -> bool:
        return self.path_for(identity_id).is_file()

    def get(self, identity_id: IdentityId) -> Optional[AvatarPack]:
        if identity_id in self._cache:
            return self._cache[identity_id]
        path = self.path_for(identity_id)
        if not path.is_file():
            return None
        pack = read_pack(path)
        self._cache[identity_id] = pack
        return pack

    def save(self, identity_id: IdentityId, pack: AvatarPack) -> AvatarIdentityHandle:
        target = self.path_for(identity_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        if pack.archive_path.resolve() != target.resolve():
            shutil.copy2(pack.archive_path, target)
        self._cache[identity_id] = AvatarPack(manifest=pack.manifest, archive_path=target)
        return AvatarIdentityHandle(
            identity_id=identity_id,
            pack_path=target,
            pack_digest=pack.digest(),
            prepared_at=pack.manifest.created_at,
        )

    def verify(self, identity_id: IdentityId) -> Tuple[bool, list]:
        path = self.path_for(identity_id)
        if not path.is_file():
            return False, ["missing pack file"]
        return len(verify_pack(path)) == 0, verify_pack(path)
