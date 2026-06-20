"""Unit tests for RedisJobRepository using a mocked Redis client.

Verifies upsert, get, and mark operations without a running Redis server
or even the ``redis`` package installed.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from contracts.job_queue import JobState, RenderJob
from src.domain.types import RenderJobId
from src.storage.jobs import RedisJobRepository


# ── helpers ─────────────────────────────────────────────────────────


def _job(
    jid: str = "job-1",
    state: JobState = JobState.PENDING,
    payload: dict | None = None,
) -> RenderJob:
    now = datetime.now(timezone.utc)
    return RenderJob(
        id=RenderJobId(jid),
        state=state,
        payload=payload or {"identity_id": "id-1", "audio_path": "/tmp/a.wav"},
        created_at=now,
        updated_at=now,
    )


def _mock_raw(mapping: dict) -> dict:
    """Return the dict that hgetall would return."""
    return mapping


def _make_repo(mock_redis: MagicMock) -> RedisJobRepository:
    """Inject a fake ``redis`` module so the import inside
    ``RedisJobRepository.__post_init__`` succeeds.
    """
    mod = MagicMock()
    mod.Redis = MagicMock()
    mod.Redis.from_url = MagicMock(return_value=mock_redis)
    with patch.dict(sys.modules, {"redis": mod}):
        return RedisJobRepository(url="redis://fake:6379/0")


# ── upsert ──────────────────────────────────────────────────────────


def test_upsert_stores_full_job_in_hash():
    r = MagicMock()
    repo = _make_repo(r)
    j = _job("job-upsert", state=JobState.RESERVED)

    repo.upsert(j)

    r.hset.assert_called_once()
    args, kwargs = r.hset.call_args
    hash_key = args[0]
    mapping = kwargs["mapping"]

    assert hash_key == "heyavatar:job:job-upsert"
    assert mapping["job_id"] == "job-upsert"
    assert mapping["state"] == "reserved"
    assert mapping["attempts"] == "0"
    assert json.loads(mapping["payload"]) == j.payload


def test_upsert_stores_json_serializable_payload():
    r = MagicMock()
    repo = _make_repo(r)
    j = _job("job-json", payload={"nested": {"key": [1, 2, 3]}})

    repo.upsert(j)

    args, kwargs = r.hset.call_args
    stored = json.loads(kwargs["mapping"]["payload"])
    assert stored == {"nested": {"key": [1, 2, 3]}}


# ── get ─────────────────────────────────────────────────────────────


def test_get_returns_none_for_missing_job():
    r = MagicMock()
    r.hgetall.return_value = {}
    repo = _make_repo(r)

    assert repo.get(RenderJobId("job-nonexistent")) is None


def test_get_reconstructs_job_from_hash():
    now = datetime.now(timezone.utc)
    r = MagicMock()
    r.hgetall.return_value = {
        "job_id": "job-get",
        "state": "running",
        "payload": json.dumps({"audio_path": "/tmp/a.wav"}),
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "attempts": "2",
        "last_error": "previous OOM",
        "reserved_by": "w-3",
    }
    repo = _make_repo(r)

    job = repo.get(RenderJobId("job-get"))

    assert job is not None
    assert job.id == RenderJobId("job-get")
    assert job.state == JobState.RUNNING
    assert job.payload == {"audio_path": "/tmp/a.wav"}
    assert job.attempts == 2
    assert job.last_error == "previous OOM"
    assert job.reserved_by == "w-3"


def test_get_handles_minimal_hash():
    r = MagicMock()
    r.hgetall.return_value = {"job_id": "job-min", "state": "completed"}
    repo = _make_repo(r)

    job = repo.get(RenderJobId("job-min"))

    assert job is not None
    assert job.state == JobState.COMPLETED
    assert job.payload == {}
    assert job.attempts == 0
    assert job.last_error is None


# ── mark ────────────────────────────────────────────────────────────


def test_mark_updates_state_and_writes_back():
    r = MagicMock()
    now = datetime.now(timezone.utc)
    r.hgetall.return_value = {
        "job_id": "job-mark",
        "state": "running",
        "payload": json.dumps({"tier": "studio"}),
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "attempts": "1",
        "last_error": "",
        "reserved_by": "w-1",
    }
    repo = _make_repo(r)

    repo.mark(RenderJobId("job-mark"), JobState.COMPLETED)

    # The final upsert should write the state as "completed".
    final_hset = r.hset.call_args_list[-1]
    assert final_hset[1]["mapping"]["state"] == "completed"


def test_mark_on_missing_job_is_noop():
    r = MagicMock()
    r.hgetall.return_value = {}
    repo = _make_repo(r)

    repo.mark(RenderJobId("job-ghost"), JobState.CANCELLED)

    # No hset calls because get returned None.
    r.hset.assert_not_called()
