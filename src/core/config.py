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

    # Distributed heartbeat — FROZEN by Change 3 / ROADMAP.md §1.
    # The MVP target is single-worker, so the heartbeat daemon
    # thread in ``workers/gpu_worker/worker.py::_start_redis_heartbeat``
    # does NOT spin up by default. Operators opting in for a future
    # multi-worker deploy set this flag via
    # ``HEYAVATAR_ENABLE_DISTRIBUTED_HEARTBEAT=1``.
    enable_distributed_heartbeat: bool = False

    # Object store backend: ``fs`` only (S3 frozen by Change 3 /
    # ROADMAP.md §1). The MVP uses the local filesystem under
    # ``HEYAVATAR_OBJECT_STORE``. S3 settings below are intentionally
    # absent from this dataclass — re-introduce them only when
    # cross-region storage becomes a real production need.
    object_store_backend: Literal["fs"] = "fs"

    # Audio-to-expression bridge backend:
    #   ``dsp``    — keep the pure-Python DSP envelope pipeline (no
    #                ML deps; default for CI to keep mock tests green).
    #   ``neural`` — require SadTalker Audio2Motion to be importable
    #                and run on GPU. If the import fails the engine
    #                transitions to DEGRADED so the orchestrator can
    #                route around the broken worker instead of
    #                silently shipping DSP-tier lip-sync to paying
    #                customers. Never auto-falls-back from ``neural``
    #                to ``dsp``; that policy is intentional.
    audio_bridge_backend: Literal["dsp", "neural"] = "dsp"

    # Distributed heartbeat — workers publish their health JSON to
    # ``heyavatar:worker:{worker_id}:health`` every
    # ``worker_health_publish_seconds`` seconds, with the Redis key
    # expiring after ``worker_pool_heartbeat_ttl`` seconds. The API
    # process polls the same key-space every
    # ``api_worker_pool_sync_seconds`` seconds so the in-process
    # WorkerPool mirrors the cluster's live capacity. All three
    # values are operator-tunable; the default TTL is 2x the publish
    # period so a transient publish miss does not yet expire the
    # record.
    worker_health_publish_seconds: float = 3.0
    api_worker_pool_sync_seconds: float = 3.0
    worker_pool_heartbeat_ttl: int = 15

    # Observability
    # OTLP exporter endpoint (gRPC), e.g. ``http://otel-collector:4317``.
    # Empty / starting with "off" disables tracing entirely so the
    # rest of the codebase can run without the SDK installed.
    otel_endpoint: str = ""
    # ``HEYAVATAR_PROCESS_ROLE`` is auto-exported as a span resource
    # attribute and a Prometheus ``process_info`` label.
    process_role: str = "api"  # overwritten to "worker" by gpu_worker.main()
    # Workers expose their Prometheus ``/metrics`` on this port. Set
    # to ``0`` to disable and rely on push-gateway instead.
    worker_metrics_port: int = 9100
    # Update ``process_info`` and route-scoped Prometheus metrics
    # even on the API process (default true).
    api_metrics_enabled: bool = True

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
            enable_distributed_heartbeat=_env_bool(
                "HEYAVATAR_ENABLE_DISTRIBUTED_HEARTBEAT", False
            ),
            audio_bridge_backend=os.environ.get(  # type: ignore[arg-type]
                "HEYAVATAR_AUDIO_BRIDGE_BACKEND", "dsp"
            ),
            worker_health_publish_seconds=_env_float(
                "HEYAVATAR_WORKER_HEALTH_PUBLISH_SECONDS", 3.0
            ),
            api_worker_pool_sync_seconds=_env_float(
                "HEYAVATAR_API_WORKER_POOL_SYNC_SECONDS", 3.0
            ),
            worker_pool_heartbeat_ttl=_env_int(
                "HEYAVATAR_WORKER_POOL_HEARTBEAT_TTL", 15
            ),
            otel_endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", ""),
            process_role=os.environ.get("HEYAVATAR_PROCESS_ROLE", "api"),
            worker_metrics_port=_env_int("HEYAVATAR_WORKER_METRICS_PORT", 9100),
            api_metrics_enabled=_env_bool("HEYAVATAR_API_METRICS_ENABLED", True),
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
