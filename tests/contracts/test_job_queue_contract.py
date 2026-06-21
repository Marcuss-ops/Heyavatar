"""Contract test for the JobQueue ABC.

Verifies that :class:`InMemoryJobQueue` implements the full
publish/reserve/ack/fail/depth contract correctly.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from contracts.job_queue import JobState, QueueHandle, RenderJob
from src.scheduler.queue.memory import InMemoryJobQueue
from src.domain.types import RenderJobId


def _job(jid: str = "job-1", payload: dict | None = None) -> RenderJob:
    now = datetime.now(timezone.utc)
    return RenderJob(
        id=RenderJobId(jid),
        state=JobState.PENDING,
        payload=payload or {"identity_id": "id-1", "audio_path": "/tmp/a.wav"},
        created_at=now,
        updated_at=now,
    )


def test_publish_then_reserve_yields_pending_job():
    q = InMemoryJobQueue()
    q.publish(_job())
    handle = QueueHandle(worker_id="w-1", engine_id="musetalk-v1", tier="any")
    job = q.reserve(handle)
    assert job is not None
    assert job.state == JobState.RESERVED
    assert job.reserved_by == "w-1"


def test_ack_marks_state_completed():
    q = InMemoryJobQueue()
    j = _job()
    q.publish(j)
    handle = QueueHandle(worker_id="w-1", engine_id="musetalk-v1", tier="any")
    reserved = q.reserve(handle)
    assert reserved is not None
    q.acknowledge(reserved.id)
    stored = q.job(reserved.id)
    assert stored is not None and stored.state == JobState.COMPLETED


def test_fail_records_reason():
    q = InMemoryJobQueue()
    j = _job()
    q.publish(j)
    handle = QueueHandle(worker_id="w-1", engine_id="musetalk-v1", tier="any")
    reserved = q.reserve(handle)
    assert reserved is not None
    q.fail(reserved.id, reason="boom")
    stored = q.job(reserved.id)
    assert stored is not None
    assert stored.state == JobState.FAILED
    assert "boom" in (stored.last_error or "")


def test_cancel_blocks_subsequent_reserve():
    q = InMemoryJobQueue()
    j = _job(jid="job-cancel")
    q.publish(j)
    q.cancel(j.id)
    handle = QueueHandle(worker_id="w-1", engine_id="musetalk-v1", tier="any")
    assert q.reserve(handle) is None


def test_depth_counts_only_pending():
    q = InMemoryJobQueue()
    assert q.depth() == 0
    q.publish(_job("job-a"))
    q.publish(_job("job-b"))
    q.publish(_job("job-c"))
    assert q.depth() == 3
    handle = QueueHandle(worker_id="w-1", engine_id="musetalk-v1", tier="any")
    job = q.reserve(handle)
    assert job is not None
    assert q.depth() == 2
