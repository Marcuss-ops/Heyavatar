"""Shared pytest marker and helper functions for full-pipeline scenario tests.

* :data:`requires_ffmpeg` — pytest marker that skips when ``ffmpeg`` is
  not on PATH. The pipeline's EncodingWorker shells out to ffmpeg for
  mp4 concatenation, so we mirror that constraint here.
* :func:`_publish_render_job` — simulates the API's ``POST /jobs``
  endpoint: builds a :class:`RenderJob` and writes it to the in-memory
  queue + repository.
* :func:`_simulate_worker_reserve` — simulates the worker's
  ``reserve()`` call and stamps the resulting state onto the repo.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from typing import Optional

import pytest

from contracts.job_queue import JobState, QueueHandle, RenderJob
from src.domain.types import RenderJobId
from src.scheduler.queue.memory import InMemoryJobQueue
from src.storage.jobs.memory import InMemoryJobRepository


requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="mock-mode E2E test shells out to ffmpeg for encoding",
)


def _publish_render_job(
    queue: InMemoryJobQueue,
    repo: InMemoryJobRepository,
    *,
    identity_id: str = "id-alice",
    source_image: str = "",
    audio_path: str = "",
    tier: str = "express",
) -> RenderJob:
    """Simulate the API's ``POST /jobs`` endpoint."""
    now = datetime.now(timezone.utc)
    job = RenderJob(
        id=RenderJobId("job-e2e-001"),
        state=JobState.PENDING,
        payload={
            "identity_id": identity_id,
            "source_image": source_image,
            "audio_path": audio_path,
            "fps": 25,
            "tier": tier,
            "job_type": "render",
        },
        created_at=now,
        updated_at=now,
    )
    queue.publish(job)
    repo.upsert(job)
    return job


def _simulate_worker_reserve(
    queue: InMemoryJobQueue,
    repo: InMemoryJobRepository,
    worker_id: str = "w-1",
) -> Optional[RenderJob]:
    """Simulate the worker's ``reserve()`` call and subsequent state update."""
    handle = QueueHandle(
        worker_id=worker_id, engine_id="musetalk-v1", tier="any"
    )
    job = queue.reserve(handle)
    if job is not None:
        repo.mark(job.id, JobState.RESERVED)
    return job
