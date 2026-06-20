"""Unit tests for RedisJobQueue using a mocked Redis client.

These tests verify the correctness of publish, reserve, acknowledge (XACK),
fail (XACK), depth (XPENDING), reclaim (XAUTOCLAIM), and trim (XTRIM)
without requiring a running Redis server or even the ``redis`` package.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from contracts.job_queue import JobState, QueueHandle, RenderJob
from src.domain.types import RenderJobId
from src.scheduler.queue import RedisJobQueue


# ── redis-module mock (needed because ``redis`` is not installed) ──

def _fake_redis_module(mock_client: MagicMock) -> MagicMock:
    """Build a fake ``redis`` module and inject into ``sys.modules``."""
    mod = MagicMock()
    mod.Redis = MagicMock()
    mod.Redis.from_url = MagicMock(return_value=mock_client)
    mod.exceptions.ResponseError = type("ResponseError", (Exception,), {})
    return mod


# ── helpers ─────────────────────────────────────────────────────────


def _job(jid: str = "job-1", payload: dict | None = None) -> RenderJob:
    now = datetime.now(timezone.utc)
    return RenderJob(
        id=RenderJobId(jid),
        state=JobState.PENDING,
        payload=payload or {"identity_id": "id-1", "audio_path": "/tmp/a.wav"},
        created_at=now,
        updated_at=now,
    )


def _handle(worker_id: str = "w-1") -> QueueHandle:
    return QueueHandle(worker_id=worker_id, engine_id="musetalk-v1", tier="any")


def _mock_redis(*, xpending_pending: int = 0) -> MagicMock:
    """Build a MagicMock that passes the Redis client constructor."""
    r = MagicMock()
    r.xpending.return_value = {"pending": xpending_pending, "consumers": []}
    r.xautoclaim.return_value = None
    r.xtrim.return_value = 0
    return r


def _make_queue(mock_redis: MagicMock) -> RedisJobQueue:
    """Inject a pre-built MagicMock as the Redis client via ``sys.modules``.

    Because the ``redis`` package is not installed in this environment,
    we inject a fake module so ``import redis`` inside
    ``RedisJobQueue.__post_init__`` succeeds.
    """
    fake_mod = _fake_redis_module(mock_redis)
    with patch.dict(sys.modules, {"redis": fake_mod}):
        return RedisJobQueue(url="redis://fake:6379/0")


# ── publish ─────────────────────────────────────────────────────────


def test_publish_calls_xadd_and_stores_payload_in_hash():
    r = _mock_redis()
    q = _make_queue(r)
    j = _job("job-x", {"foo": "bar"})

    q.publish(j)

    # Side-hash stores payload + state.
    r.hset.assert_called_once()
    args, kwargs = r.hset.call_args
    # First positional arg is the hash key.
    assert "heyavatar:job:job-x" in args[0]
    # Stream entry is created.
    r.xadd.assert_called_once_with(
        "heyavatar:jobs",
        {"job_id": "job-x", "state": "pending"},
    )


# ── reserve ─────────────────────────────────────────────────────────


def test_reserve_runs_housekeeping_once_then_skips():
    r = _mock_redis()
    r.xreadgroup.return_value = None  # no messages
    r.xpending.return_value = {"pending": 0}
    q = _make_queue(r)

    assert q.reserve(_handle()) is None
    assert q.reserve(_handle()) is None

    # Housekeeping (reclaim + trim) called only on first reserve.
    assert r.xautoclaim.call_count == 1
    assert r.xtrim.call_count == 1


def test_reserve_reconstructs_job_from_side_hash():
    r = _mock_redis()
    r.xreadgroup.return_value = [
        ("heyavatar:jobs", [("1234567890123-0", {"job_id": "job-a"})])
    ]
    r.hget.return_value = json.dumps({"audio_path": "/tmp/a.wav", "tier": "express"})
    q = _make_queue(r)

    job = q.reserve(_handle("w-2"))

    assert job is not None
    assert job.id == RenderJobId("job-a")
    assert job.state == JobState.RESERVED
    assert job.payload == {"audio_path": "/tmp/a.wav", "tier": "express"}
    assert job.reserved_by == "w-2"

    # Stream entry id stored in side-hash for later XACK.
    r.hset.assert_any_call(
        "heyavatar:job:job-a",
        mapping={
            "_stream_entry": "1234567890123-0",
            "state": "reserved",
            "reserved_by": "w-2",
        },
    )


def test_reserve_returns_none_when_stream_empty():
    r = _mock_redis()
    r.xreadgroup.return_value = None
    q = _make_queue(r)

    assert q.reserve(_handle()) is None


def test_reserve_handles_empty_entries():
    r = _mock_redis()
    r.xreadgroup.return_value = [("heyavatar:jobs", [])]
    q = _make_queue(r)

    assert q.reserve(_handle()) is None


# ── acknowledge (XACK) ──────────────────────────────────────────────


def test_acknowledge_calls_xack_and_cleans_entry_marker():
    r = _mock_redis()
    r.hget.return_value = "1234567890123-0"
    q = _make_queue(r)

    q.acknowledge(RenderJobId("job-ack"))

    # State updated to completed.
    r.hset.assert_called_with(
        "heyavatar:job:job-ack",
        mapping={"state": "completed"},
    )
    # XACK called with the stream entry id.
    r.xack.assert_called_once_with(
        "heyavatar:jobs",
        "heyavatar-workers",
        "1234567890123-0",
    )
    # Entry marker cleaned.
    r.hdel.assert_called_once_with(
        "heyavatar:job:job-ack",
        "_stream_entry",
    )


def test_acknowledge_skips_xack_when_no_entry_marker():
    r = _mock_redis()
    r.hget.return_value = None  # no stream entry stored
    q = _make_queue(r)

    q.acknowledge(RenderJobId("job-no-ack"))

    r.hset.assert_called_once()
    r.xack.assert_not_called()
    r.hdel.assert_not_called()


# ── fail (XACK) ─────────────────────────────────────────────────────


def test_fail_calls_xack_and_stores_error():
    r = _mock_redis()
    r.hget.return_value = "9999999999999-0"
    q = _make_queue(r)

    q.fail(RenderJobId("job-fail"), reason="GPU OOM")

    r.hset.assert_called_with(
        "heyavatar:job:job-fail",
        mapping={"state": "failed", "error": "GPU OOM"},
    )
    r.xack.assert_called_once_with(
        "heyavatar:jobs",
        "heyavatar-workers",
        "9999999999999-0",
    )
    r.hdel.assert_called_once_with(
        "heyavatar:job:job-fail",
        "_stream_entry",
    )


def test_fail_skips_xack_when_no_entry_marker():
    r = _mock_redis()
    r.hget.return_value = None
    q = _make_queue(r)

    q.fail(RenderJobId("job-fail2"), reason="timeout")

    r.hset.assert_called_once()
    r.xack.assert_not_called()


# ── depth (XPENDING) ────────────────────────────────────────────────


def test_depth_returns_xpending_count():
    r = _mock_redis(xpending_pending=7)
    q = _make_queue(r)

    assert q.depth() == 7

    # XPENDING was queried, not XLEN.
    r.xpending.assert_called_once_with("heyavatar:jobs", "heyavatar-workers")
    r.xlen.assert_not_called()


def test_depth_returns_zero_on_exception():
    r = _mock_redis()
    r.xpending.side_effect = RuntimeError("stream gone")
    q = _make_queue(r)

    assert q.depth() == 0


# ── reclaim (XAUTOCLAIM) ────────────────────────────────────────────


def test_reclaim_xacks_abandoned_messages():
    r = _mock_redis()
    q = _make_queue(r)

    # First call returns abandoned messages; second returns empty.
    r.xautoclaim.side_effect = [
        ("9999999999999-1", [("1234567890123-0", {"job_id": "job-orphan"})], []),
        None,
    ]

    claimed = q._reclaim()

    assert claimed == 1
    # Job hash updated to failed with reason.
    r.hset.assert_any_call(
        "heyavatar:job:job-orphan",
        mapping={
            "state": "failed",
            "error": "abandoned: worker timed out",
        },
    )
    # Abandoned message XACK'd.
    r.xack.assert_any_call(
        "heyavatar:jobs",
        "heyavatar-workers",
        "1234567890123-0",
    )


def test_reclaim_returns_zero_when_nothing_abandoned():
    r = _mock_redis()
    r.xautoclaim.return_value = ("0-0", [], [])
    q = _make_queue(r)

    assert q._reclaim() == 0


def test_reclaim_handles_xautoclaim_exception_gracefully():
    r = _mock_redis()
    r.xautoclaim.side_effect = RuntimeError("redis down")
    q = _make_queue(r)

    assert q._reclaim() == 0


# ── trim (XTRIM) ────────────────────────────────────────────────────


def test_trim_calls_xtrim_with_approximate():
    r = _mock_redis()
    r.xtrim.return_value = 42
    q = _make_queue(r)

    trimmed = q._trim(maxlen=5_000)

    assert trimmed == 42
    r.xtrim.assert_called_once_with(
        "heyavatar:jobs",
        maxlen=5_000,
        approximate=True,
    )


def test_trim_returns_zero_on_exception():
    r = _mock_redis()
    r.xtrim.side_effect = RuntimeError("redis down")
    q = _make_queue(r)

    assert q._trim() == 0


# ── cancel ──────────────────────────────────────────────────────────


def test_reserve_with_missing_payload_returns_empty_dict():
    r = _mock_redis()
    r.xreadgroup.return_value = [
        ("heyavatar:jobs", [("1234567890123-0", {"job_id": "job-nopayload"})])
    ]
    r.hget.return_value = None  # side-hash never written
    q = _make_queue(r)

    job = q.reserve(_handle())

    assert job is not None
    assert job.payload == {}


def test_acknowledge_twice_is_idempotent():
    r = _mock_redis()
    # First call: entry marker present.
    r.hget.return_value = "entry-1"
    q = _make_queue(r)

    q.acknowledge(RenderJobId("job-twice"))
    assert r.xack.call_count == 1
    assert r.hdel.call_count == 1

    # Second call: entry already cleaned, no XACK.
    r.reset_mock()
    r.hget.return_value = None
    q.acknowledge(RenderJobId("job-twice"))
    r.xack.assert_not_called()
    r.hdel.assert_not_called()


def test_cancel_sets_cancelled_state_in_hash():
    r = _mock_redis()
    q = _make_queue(r)

    q.cancel(RenderJobId("job-cancel"))

    r.hset.assert_called_once_with(
        "heyavatar:job:job-cancel",
        mapping={"state": "cancelled"},
    )
