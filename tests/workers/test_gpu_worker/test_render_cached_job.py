"""Tests for the GPU worker's ``job_type=render_cached`` dispatcher branch.

Three scenarios:

1. Happy path — payload carries avatar_id + gesture_id + identity_id +
   audio_path; the worker calls :func:`render_cached_avatar` (mocked) and
   returns ``JobState.COMPLETED`` plus the canonical metrics block.
2. QC failure — the use case reports ``passed=False``; worker surfaces
   ``JobState.COMPLETED_DEGRADED`` with ``degraded=True`` and the QC
   verdict in ``result["qc_status"]``.
3. Real-mode engine failure — :func:`render_cached_avatar` raises
   ``RuntimeError``; worker returns ``JobState.FAILED_INFERENCE`` with
   ``result["error"]`` carrying the engine's message.

A fourth test confirms ``JobSubmitRequest`` (the API validator)
rejects ``job_type=render_cached`` payloads that miss
``avatar_id`` / ``gesture_id`` with a 422-shaped ``ValueError``.
"""

from __future__ import annotations

import math
import shutil
import struct
import wave
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from api.schemas.jobs import JobSubmitRequest
from contracts.job_queue import JobState, RenderJob
from contracts.quality_checker import QCResult
from providers import get_provider
from src.application.render_cached_avatar import RenderCachedAvatarResult
from src.core.config import get_settings
from src.domain.enums import EngineId
from src.domain.types import IdentityId
from src.storage.avatar_packs import AvatarPackRepository
from tests._fixtures import PNG_1X1 as _PNG_1x1
from tests.domain.test_body_template import _make_synthetic_template


from tests.workers.test_gpu_worker._helpers import (
    _build_worker,
    requires_ffmpeg,
)


# ─── helpers ────────────────────────────────────────────────────────────────


def _make_wav(path: Path, seconds: float = 2.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000
    n = int(seconds * sample_rate)
    samples = []
    for i in range(n):
        if i < sample_rate * 0.1 or i > sample_rate * 1.9:
            samples.append(0)
        else:
            samples.append(int(0.6 * 32767 * math.sin(2 * math.pi * 220 * (i / sample_rate))))
    raw = struct.pack("<" + "h" * len(samples), *samples)
    with wave.open(str(path), "wb") as wh:
        wh.setnchannels(1)
        wh.setsampwidth(2)
        wh.setframerate(sample_rate)
        wh.writeframes(raw)
    return path


def _make_render_cached_job(
    *,
    job_id: str,
    avatar_id: str,
    gesture_id: str,
    identity_id: str = "id-cached",
    audio_path: str = "",
    source_image: str = "",
    body_templates_dir: str = "",
) -> RenderJob:
    now = datetime.now(timezone.utc)
    return RenderJob(
        id=job_id,
        state=JobState.RUNNING,
        payload={
            "job_type": "render_cached",
            "identity_id": identity_id,
            "source_image": source_image,
            "audio_path": audio_path,
            "avatar_id": avatar_id,
            "gesture_id": gesture_id,
            "fps": 25,
            "tier": "express",
        },
        created_at=now,
        updated_at=now,
    )


def _stub_cached_result(
    *,
    tmp_path: Path,
    job_id: str,
    qc_passed: bool = True,
    gpu_seconds: float = 0.4,
    wall_seconds: float = 1.2,
    body_cache_hit: bool = True,
    identity_cache_hit: bool = False,
):
    """Build a :class:`RenderCachedAvatarResult` loaded with the canonical metrics block."""
    qc_status = "COMPLETED" if qc_passed else "FAILED_QC_DEBUG_OVERLAY"
    qc_result = QCResult(
        passed=qc_passed,
        status=qc_status,
        debug_green_ratio=0.0 if qc_passed else 0.05,
        black_frame_ratio=0.0,
        duration_delta_ms=0.0,
        frames_expected=50,
        frames_actual=50,
        invalid_transforms=0,
        errors=[] if qc_passed else ["debug green overlay detected"],
    )
    final_path = tmp_path / job_id / "final.mp4"
    composited = tmp_path / job_id / "composited.mp4"
    return RenderCachedAvatarResult(
        status=qc_status,
        avatar_id="avatar_001",
        gesture_id="explain_both",
        face_roi_path=tmp_path / job_id / "face_roi.mp4",
        face_lipsynced_path=tmp_path / job_id / "face_lipsynced.mp4",
        composited_path=composited,
        final_path=final_path if qc_passed else None,
        body_dir=tmp_path / "body_templates" / "avatar_001" / "explain_both",
        output_seconds=2.0,
        wall_seconds=wall_seconds,
        gpu_seconds=gpu_seconds,
        face_resolution=(256, 256),
        batch_size=8,
        body_cache_hit=body_cache_hit,
        identity_cache_hit=identity_cache_hit,
        model_warm=True,
        face_region_only=True,
        qc_result=qc_result,
        metrics={
            "status": qc_status,
            "avatar_id": "avatar_001",
            "gesture_id": "explain_both",
            "identity_id": "id-cached",
            "output_seconds": 2.0,
            "wall_seconds": wall_seconds,
            "gpu_seconds": gpu_seconds,
            "gpu_seconds_per_output_minute": gpu_seconds * 60.0 / 2.0,
            "face_resolution": [256, 256],
            "batch_size": 8,
            "body_cache_hit": body_cache_hit,
            "identity_cache_hit": identity_cache_hit,
            "model_warm": True,
            "face_region_only": True,
            "qc": {
                "debug_green_ratio": 0.0,
                "black_frame_ratio": 0.0,
                "duration_delta_ms": 0.0,
                "frames_expected": 50,
                "frames_actual": 50,
            },
        },
    )


# ─── tests ──────────────────────────────────────────────────────────────────


@requires_ffmpeg
def test_do_process_render_cached_completed(workdir, tmp_path):
    """QC passed → JobState.COMPLETED with metrics block in result."""
    source = tmp_path / "alice.png"
    source.write_bytes(_PNG_1x1)
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 200 + b"WAVE")

    engine = get_provider(EngineId.MUSE_TALK)
    engine.load()
    try:
        pack_repo = AvatarPackRepository(root=workdir / "packs")
        worker = _build_worker(pack_repo=pack_repo)
        job = _make_render_cached_job(
            job_id="job-cached-happy",
            avatar_id="avatar_001",
            gesture_id="explain_both",
            identity_id="id-cached",
            audio_path=str(audio.resolve()),
            source_image=str(source.resolve()),
            body_templates_dir=str(tmp_path),
        )
        stub = _stub_cached_result(
            tmp_path=tmp_path, job_id=str(job.id), qc_passed=True,
        )
        with patch(
            "workers.gpu_worker.process.render_cached_avatar",
            return_value=stub,
        ) as mock_called:
            state, result = worker._do_process_render_cached(job, engine, pack_repo)

        assert mock_called.called, "render_cached_avatar must be invoked"
        kwargs = mock_called.call_args.kwargs
        assert kwargs["avatar_id"] == "avatar_001"
        assert kwargs["gesture_id"] == "explain_both"
        assert kwargs["identity_id"] == "id-cached"
        assert kwargs["engine"] is engine

        assert state == JobState.COMPLETED
        assert result["degraded"] is False
        assert result["qc_status"] == "COMPLETED"
        assert result["output_path"] == str(stub.final_path)
        assert result["duration_seconds"] == pytest.approx(2.0)
        assert result["gpu_seconds"] == pytest.approx(0.4)
        assert result["wall_seconds"] == pytest.approx(1.2)
        assert result["face_resolution"] == [256, 256]
        assert result["batch_size"] == 8
        # Full metrics block survives the round-trip.
        assert result["metrics"]["body_cache_hit"] is True
        assert result["metrics"]["identity_cache_hit"] is False
    finally:
        engine.unload()


@requires_ffmpeg
def test_do_process_render_cached_qc_failed(workdir, tmp_path):
    """QC rejection → JobState.COMPLETED_DEGRADED with the QC verdict in metrics."""
    source = tmp_path / "alice.png"
    source.write_bytes(_PNG_1x1)
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 200 + b"WAVE")

    engine = get_provider(EngineId.MUSE_TALK)
    engine.load()
    try:
        pack_repo = AvatarPackRepository(root=workdir / "packs")
        worker = _build_worker(pack_repo=pack_repo)
        job = _make_render_cached_job(
            job_id="job-cached-qc-failed",
            avatar_id="avatar_001",
            gesture_id="explain_both",
            audio_path=str(audio.resolve()),
            source_image=str(source.resolve()),
        )
        stub = _stub_cached_result(
            tmp_path=tmp_path, job_id=str(job.id), qc_passed=False,
        )
        with patch(
            "workers.gpu_worker.process.render_cached_avatar",
            return_value=stub,
        ):
            state, result = worker._do_process_render_cached(job, engine, pack_repo)

        assert state == JobState.COMPLETED_DEGRADED
        assert result["degraded"] is True
        assert result["qc_status"] == "FAILED_QC_DEBUG_OVERLAY"
    finally:
        engine.unload()


@requires_ffmpeg
def test_do_process_render_cached_engine_failure_returns_failed_inference(
    workdir, tmp_path
):
    """render_cached_avatar raises RuntimeError → JobState.FAILED_INFERENCE."""
    source = tmp_path / "alice.png"
    source.write_bytes(_PNG_1x1)
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 200 + b"WAVE")

    engine = get_provider(EngineId.MUSE_TALK)
    engine.load()
    try:
        pack_repo = AvatarPackRepository(root=workdir / "packs")
        worker = _build_worker(pack_repo=pack_repo)
        job = _make_render_cached_job(
            job_id="job-cached-engine-down",
            avatar_id="avatar_001",
            gesture_id="explain_both",
            audio_path=str(audio.resolve()),
            source_image=str(source.resolve()),
        )
        with patch(
            "workers.gpu_worker.process.render_cached_avatar",
            side_effect=RuntimeError("simulated upstream VAE unavailable"),
        ):
            state, result = worker._do_process_render_cached(job, engine, pack_repo)

        assert state == JobState.FAILED_INFERENCE
        assert result["degraded"] is True
        assert "simulated upstream VAE unavailable" in result["error"]
        assert result["failed_stage"] == "engine"
        assert result["output_path"] is None
    finally:
        engine.unload()


def test_do_process_render_cached_missing_avatar_id_raises(workdir, tmp_path):
    """Worker-side guardrail: payload missing avatar_id/gesture_id raises."""
    source = tmp_path / "alice.png"
    source.write_bytes(_PNG_1x1)
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 200 + b"WAVE")

    engine = get_provider(EngineId.MUSE_TALK)
    engine.load()
    try:
        pack_repo = AvatarPackRepository(root=workdir / "packs")
        worker = _build_worker(pack_repo=pack_repo)
        job = RenderJob(
            id="job-bad-cached",
            state=JobState.RUNNING,
            payload={
                "job_type": "render_cached",
                "identity_id": "id-x",
                "source_image": str(source.resolve()),
                "audio_path": str(audio.resolve()),
                "fps": 25,
                "tier": "express",
                # avatar_id / gesture_id missing on purpose.
            },
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        with pytest.raises(RuntimeError) as exc:
            worker._do_process_render_cached(job, engine, pack_repo)
        msg = str(exc.value)
        assert "avatar_id" in msg and "gesture_id" in msg
    finally:
        engine.unload()


# ─── API validator ──────────────────────────────────────────────────────────


def test_job_submit_request_rejects_render_cached_without_avatar(tmp_path):
    """API layer rejects render_cached jobs that miss body-template keys."""
    with pytest.raises(ValidationError) as exc:
        JobSubmitRequest(
            identity_id="id-x",
            source_image=str(tmp_path / "alice.png"),
            audio_path=str(tmp_path / "speech.wav"),
            job_type="render_cached",
            # avatar_id / gesture_id missing on purpose.
        )
    msg = str(exc.value)
    assert "avatar_id" in msg and "gesture_id" in msg


def test_job_submit_request_render_cached_carries_avatar_keys_through_to_queue_payload(tmp_path):
    """When both keys are present, ``to_queue_payload`` propagates them."""
    req = JobSubmitRequest(
        identity_id="id-x",
        source_image=str(tmp_path / "alice.png"),
        audio_path=str(tmp_path / "speech.wav"),
        job_type="render_cached",
        avatar_id="avatar_001",
        gesture_id="explain_both",
    )
    payload = req.to_queue_payload()
    assert payload["job_type"] == "render_cached"
    assert payload["avatar_id"] == "avatar_001"
    assert payload["gesture_id"] == "explain_both"


def test_job_submit_request_default_is_render_job_type(tmp_path):
    """Default behavior preserved — existing callers see ``job_type='render'``."""
    req = JobSubmitRequest(
        identity_id="id-x",
        source_image=str(tmp_path / "alice.png"),
        audio_path=str(tmp_path / "speech.wav"),
    )
    assert req.job_type == "render"
    assert req.avatar_id is None
    assert req.gesture_id is None
    # The default validation must NOT raise — i.e. legacy callers
    # keep passing through the API unchanged.
    req.to_queue_payload()
