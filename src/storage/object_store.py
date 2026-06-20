"""Object store abstraction.

The API, the workers, and the encoder all exchange large blobs (avatar
packs, rendered chunks, final videos). v1 ships a file-system
implementation; production deployments swap to S3 / MinIO via the same
interface.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Optional

from src.core.config import Settings, get_settings
from src.domain.types import BucketKey


class ObjectStore(abc.ABC):
    """Abstract large-blob storage."""

    @abc.abstractmethod
    def put(self, key: BucketKey, data: BinaryIO | bytes) -> int: ...

    @abc.abstractmethod
    def get_path(self, key: BucketKey) -> Path: ...

    @abc.abstractmethod
    def exists(self, key: BucketKey) -> bool: ...

    @abc.abstractmethod
    def remove(self, key: BucketKey) -> None: ...


@dataclass(slots=True)
class FsObjectStore(ObjectStore):
    """Filesystem-backed object store. Keys are mapped under ``root``."""

    root: Path
    _bound: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, key: BucketKey, data: BinaryIO | bytes) -> int:
        target = self.get_path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, (bytes, bytearray)):
            target.write_bytes(bytes(data))
            return len(data)
        size = 0
        with target.open("wb") as fh:
            while True:
                chunk = data.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                size += len(chunk)
        return size

    def get_path(self, key: BucketKey) -> Path:
        return self.root / key

    def exists(self, key: BucketKey) -> bool:
        return self.get_path(key).is_file()

    def remove(self, key: BucketKey) -> None:
        path = self.get_path(key)
        if path.is_file():
            path.unlink()


def build_object_store(settings: Optional[Settings] = None) -> ObjectStore:
    """Build the object store whose backend is configured in settings."""
    settings = settings or get_settings()
    if settings.object_store_backend == "fs":
        return FsObjectStore(root=settings.object_store_root)
    raise NotImplementedError(
        f"Object store backend '{settings.object_store_backend}' is not yet implemented in v1. "
        "Add an S3-backed ObjectStore subclass to wire it up."
    )
