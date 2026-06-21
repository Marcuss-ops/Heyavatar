"""Low-cardinality label sets for Prometheus metrics.

These tuples are the contract for what is allowed in label values.
Adding any high-cardinality (job_id, identity_id, request path, etc.)
identifier here is a contract violation.
"""

from __future__ import annotations

from typing import Dict


ENGINE_IDS = ("musetalk-v1", "liveportrait-human-v1", "echomimic-v1", "mock")
TIERS = ("express", "studio", "premium")
JOB_STATES = ("pending", "reserved", "rendering", "completed", "failed", "cancelled")
WORKER_STATES = ("unloaded", "loading", "idle", "rendering", "degraded", "error")
QUEUE_BACKENDS = ("memory", "null", "redis")
HTTP_METHODS = ("GET", "POST", "DELETE", "PUT", "PATCH")


def _labels_kwargs(engine_id: str, tier: str) -> Dict[str, str]:
    return {"engine_id": engine_id, "tier": tier}
