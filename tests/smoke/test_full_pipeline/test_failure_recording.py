"""Failure path test: failed jobs persisted in queue + repo."""

from __future__ import annotations

from pathlib import Path

from contracts.job_queue import JobState
from src.scheduler.queue.memory import InMemoryJobQueue
from src.storage.jobs.memory import InMemoryJobRepository

from tests.smoke.test_full_pipeline._helpers import (
    _publish_render_job,
    _simulate_worker_reserve,
)


def test_full_pipeline_failed_job_is_recorded(workdir, tmp_path):
    """Verify that a failed job is correctly recorded in queue + repo."""
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 200 + b"WAVE")

    queue = InMemoryJobQueue()
    repo = InMemoryJobRepository()

    job = _publish_render_job(
        queue,
        repo,
        identity_id="id-nonexistent",
        source_image=str(tmp_path / "missing.png"),
        audio_path=str(audio.resolve()),
    )
    assert repo.get(job.id).state == JobState.PENDING

    reserved = _simulate_worker_reserve(queue, repo)
    assert reserved is not None
    assert repo.get(job.id).state == JobState.RESERVED

    # Simulate worker failure.
    queue.fail(job.id, reason="GPU OOM during render")
    repo.mark(job.id, JobState.FAILED, error="GPU OOM during render")

    final = repo.get(job.id)
    assert final is not None
    assert final.state == JobState.FAILED
    assert "GPU OOM" in (final.last_error or "")
    assert queue.depth() == 0
