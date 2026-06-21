"""API tests for the jobs endpoints.

Verifies that ``GET /jobs/{job_id}`` returns the result fields
(``result_url``, ``identity_id``, ``degraded``) after a worker
has completed the job and stored its result.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app.factory import create_app
from contracts.job_queue import JobState, RenderJob
from src.domain.types import RenderJobId
from tests._fixtures import PNG_1X1 as _PNG_1x1


@pytest.fixture
def client(workdir):
    """FastAPI TestClient with mock settings."""
    app = create_app()
    with TestClient(app) as c:
        yield c


def _make_result(*, identity_id: str, output_path: str, degraded: bool) -> dict:
    return {
        "identity_id": identity_id,
        "output_path": output_path,
        "engine_id": "musetalk-v1",
        "duration_seconds": 4.0,
        "gpu_seconds": 0.5,
        "degraded": degraded,
        "degraded_chunks": [1] if degraded else [],
        "total_chunks": 2 if degraded else 1,
    }


def _api_key_headers() -> dict:
    """Headers for API-key auth (dev mode passes through)."""
    return {"X-API-Key": "dev-mode"}


def test_get_job_returns_result_url_identity_id_and_degraded(client, tmp_path):
    """Submit render job, store a completed result, GET returns result fields."""
    source = tmp_path / "alice.png"
    source.write_bytes(_PNG_1x1)
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 200 + b"WAVE")

    # ── 1. Submit a render job ──────────────────────────────────
    submit_payload = {
        "identity_id": "id-alice",
        "source_image": str(source.resolve()),
        "audio_path": str(audio.resolve()),
        "fps": 25,
        "tier": "express",
    }
    resp = client.post("/jobs", json=submit_payload, headers=_api_key_headers())
    assert resp.status_code == 202, resp.text
    data = resp.json()
    job_id = data["job_id"]
    assert data["state"] == "pending"

    # ── 2. Simulate worker completing the job with a result ─────
    state = client.app.state.deps
    repo = state.job_repo
    fake_output = tmp_path / f"{job_id}.mp4"
    fake_output.write_bytes(b"mock-mp4")

    repo.mark(
        RenderJobId(job_id),
        JobState.COMPLETED,
        result=_make_result(
            identity_id="id-alice",
            output_path=str(fake_output.resolve()),
            degraded=False,
        ),
    )

    # ── 3. GET /jobs/{job_id} ───────────────────────────────────
    resp = client.get(f"/jobs/{job_id}", headers=_api_key_headers())
    assert resp.status_code == 200, resp.text
    job = resp.json()

    assert job["job_id"] == job_id
    assert job["state"] == "completed"
    assert job["identity_id"] == "id-alice"
    assert job["result_url"] is not None, "result_url should be populated"
    assert job["result_url"].startswith("/captures/")
    assert job["output_path"] is not None
    assert job["degraded"] is False
    assert job["engine_id"] == "musetalk-v1"
    assert job["duration_seconds"] == 4.0
    assert job["gpu_seconds"] == 0.5


def test_get_job_returns_degraded_true(client, tmp_path):
    """Submit render job with a COMPLETED_DEGRADED result → degraded=true."""
    source = tmp_path / "alice.png"
    source.write_bytes(_PNG_1x1)
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 200 + b"WAVE")

    # ── 1. Submit ───────────────────────────────────────────────
    resp = client.post(
        "/jobs",
        json={
            "identity_id": "id-alice",
            "source_image": str(source.resolve()),
            "audio_path": str(audio.resolve()),
        },
        headers=_api_key_headers(),
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    # ── 2. Store degraded result ────────────────────────────────
    state = client.app.state.deps
    fake_output = tmp_path / f"{job_id}.mp4"
    fake_output.write_bytes(b"mock-mp4")

    state.job_repo.mark(
        RenderJobId(job_id),
        JobState.COMPLETED_DEGRADED,
        result=_make_result(
            identity_id="id-alice",
            output_path=str(fake_output.resolve()),
            degraded=True,
        ),
    )

    # ── 3. GET ──────────────────────────────────────────────────
    resp = client.get(f"/jobs/{job_id}", headers=_api_key_headers())
    assert resp.status_code == 200
    job = resp.json()

    assert job["state"] == "completed_degraded"
    assert job["degraded"] is True
    assert job["identity_id"] == "id-alice"
    assert job["result_url"] is not None


def test_get_job_returns_failed_encoding(client, tmp_path):
    """Submit render job with FAILED_ENCODING → degraded=False, error in result."""
    source = tmp_path / "alice.png"
    source.write_bytes(_PNG_1x1)
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 200 + b"WAVE")

    resp = client.post(
        "/jobs",
        json={
            "identity_id": "id-alice",
            "source_image": str(source.resolve()),
            "audio_path": str(audio.resolve()),
        },
        headers=_api_key_headers(),
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    state = client.app.state.deps
    state.job_repo.mark(
        RenderJobId(job_id),
        JobState.FAILED_ENCODING,
        result={
            "identity_id": "id-alice",
            "engine_id": "musetalk-v1",
            "duration_seconds": 2.0,
            "gpu_seconds": 0.3,
            "degraded": False,
            "degraded_chunks": [],
            "total_chunks": 1,
            "error": "ffmpeg: h264_nvenc encoder not found",
        },
    )

    resp = client.get(f"/jobs/{job_id}", headers=_api_key_headers())
    assert resp.status_code == 200
    job = resp.json()

    assert job["state"] == "failed_encoding"
    assert job["last_error"] is None  # last_error is separate from result.error
    assert job["identity_id"] == "id-alice"
    assert job["output_path"] is None  # encoding failed, no output


def test_get_nonexistent_job_returns_404(client):
    """GET /jobs/nonexistent → 404."""
    resp = client.get("/jobs/job-nonexistent", headers=_api_key_headers())
    assert resp.status_code == 404
