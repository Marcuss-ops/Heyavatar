"""Environment-driven settings for Heyavatar.

Centralising env-var parsing here keeps workers, API, and adapters
consistent and avoids subtle diverges (e.g. one process reading
``REDIS_URL`` and another reading ``HEYAVATAR_REDIS_URL``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

DEFAULT_PACK_DIR = Path("./avatar_packs")
DEFAULT_CAPTURE_DIR = Path("./captures")


@dataclass(slots=True, frozen=True)
class Settings:
    """Process-wide configuration loaded from environment variables.

    Use :func:`get_settings` to obtain a memoised instance rather than
    constructing ``Settings(...)`` directly.
    """

    # Logging
    log_level: LogLevel = "INFO"
    log_json: bool = False

    # Mock mode — set ``HEYAVATAR_MOCK_ENGINE=1`` and every adapter will
    # run its full pipeline in deterministic CPU mode. Crucial for CI.
    mock_engine: bool = False

    # Paths
    pack_dir: Path = DEFAULT_PACK_DIR
    capture_dir: Path = DEFAULT_CAPTURE_DIR
    registry_file: Path = Path("registry/models.yaml")
    object_store_root: Path = Path("./object_store")

    # Queue backend: ``null`` (no-op), ``redis`` (Streams), ``memory`` (single-process only)
    queue_backend: Literal["null", "redis", "memory"] = "memory"
    redis_url: Optional[str] = None
    job_poll_interval_seconds: float = 0.5
    job_visibility_timeout_seconds: int = 120

    # Worker behaviour
    worker_id: str = "worker-001"
    worker_engine_id: str = "musetalk-v1"
    worker_check_cancel_every_n_frames: int = 8

    # Object store backend: ``fs`` (default) or ``s3``
    object_store_backend: Literal["fs", "s3"] = "fs"
    s3_endpoint_url: Optional[str] = None
    s3_bucket: Optional[str] = None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            log_level=os.environ.get("HEYAVATAR_LOG_LEVEL", "INFO"),
            log_json=_env_bool("HEYAVATAR_LOG_JSON", False),
            mock_engine=_env_bool("HEYAVATAR_MOCK_ENGINE", False),
            pack_dir=Path(os.environ.get("HEYAVATAR_PACK_DIR", str(DEFAULT_PACK_DIR))),
            capture_dir=Path(os.environ.get("HEYAVATAR_CAPTURE_DIR", str(DEFAULT_CAPTURE_DIR))),
            registry_file=Path(os.environ.get("HEYAVATAR_REGISTRY", "registry/models.yaml")),
            object_store_root=Path(os.environ.get("HEYAVATAR_OBJECT_STORE", "./object_store")),
            queue_backend=os.environ.get("HEYAVATAR_QUEUE_BACKEND", "memory"),
            redis_url=os.environ.get("REDIS_URL"),
            job_poll_interval_seconds=_env_float("HEYAVATAR_JOB_POLL_INTERVAL", 0.5),
            job_visibility_timeout_seconds=_env_int("HEYAVATAR_VISIBILITY_TIMEOUT", 120),
            worker_id=os.environ.get("HEYAVATAR_WORKER_ID", "worker-001"),
            worker_engine_id=os.environ.get("HEYAVATAR_WORKER_ENGINE", "musetalk-v1"),
            worker_check_cancel_every_n_frames=_env_int("HEYAVATAR_CANCEL_CHECK_EVERY", 8),
            object_store_backend=os.environ.get("HEYAVATAR_OBJECT_STORE_BACKEND", "fs"),
            s3_endpoint_url=os.environ.get("HEYAVATAR_S3_ENDPOINT"),
            s3_bucket=os.environ.get("HEYAVATAR_S3_BUCKET"),
        )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide memoised settings instance."""
    return Settings.from_env()
