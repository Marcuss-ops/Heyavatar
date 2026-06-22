"""Tests for :func:`src.application.render_cached_avatar.render_cached_avatar`.

Covers the four doors the plan calls out:

1. Happy path with mock MuseTalk engine — produce face_roi, face_lips,
   composited, final mp4s and the QC verdict.
2. Body-template fail-closed — missing files raise :class:`FileNotFoundError`.
3. Identity cache hit — pre-built pack short-circuits the compile step.
4. Metrics persistence — ``metrics.json`` carries the canonical block-1
   schema (``gpu_seconds``, ``body_cache_hit``, ``identity_cache_hit``,
   ``face_region_only``, ``batch_size``, …).

The tests run in mock mode (``HEYAVATAR_MOCK_ENGINE=1`` forced by
:file:`tests/conftest.py`) so no GPU / weights are needed.
"""

from __future__ import annotations

import json
import math
import shutil
import struct
import wave
from pathlib import Path
from typing import Iterator

import pytest

from src.application.render_cached_avatar import (
    RenderCachedAvatarResult,
    render_cached_avatar,
)
from tests._fixtures import PNG_1X1 as _PNG_1x1
from tests.domain.test_body_template import _make_synthetic_template


# ─── audio stub ──────────────────────────────────────────────────────────────


def _make_wav(path: Path, seconds: float = 2.0, freq: float = 220.0, sample_rate: int = 16000) -> Path:
    """Deterministic 2 s WAV so ffprobe reports a known duration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(seconds * sample_rate)
    samples = []
    for i in range(n):
        if i < sample_rate * 0.1 or i > sample_rate * 1.9:
            samples.append(0)
        else:
            samples.append(int(0.6 * 32767 * math.sin(2 * math.pi * freq * (i / sample_rate))))
    raw = struct.pack("<" + "h" * len(samples), *samples)
    with wave.open(str(path), "wb") as wh:
        wh.setnchannels(1)
        wh.setsampwidth(2)
        wh.setframerate(sample_rate)
        wh.writeframes(raw)
    return path


def _write_source_png(tmp_path: Path) -> Path:
    """Write a 1×1 PNG to disk so the (mock) engine's seed_from_path works."""
    src = tmp_path / "face.png"
    src.write_bytes(_PNG_1x1)
    return src


# ─── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def body_template(tmp_path: Path) -> Iterator[Path]:
    """Materialise a 5-file body template at ``tmp_path/avatar_001/explain_both``."""
    _make_synthetic_template(tmp_path, "avatar_001", "explain_both", total_frames=75)
    yield tmp_path


@pytest.fixture
def audio_path(tmp_path: Path) -> Path:
    return _make_wav(tmp_path / "speech.wav", seconds=2.0)


@pytest.fixture
def mock_engine():
    """A :class:`MuseTalkAdapter` instance loaded in mock mode (set by conftest)."""
    from providers.musetalk.adapter.engine import MuseTalkAdapter
    engine = MuseTalkAdapter()
    engine.load()
    try:
        yield engine
    finally:
        engine.unload()


requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="render_cached_avatar shells out to ffmpeg",
)


# ─── tests ───────────────────────────────────────────────────────────────────


@requires_ffmpeg
def test_render_cached_avatar_happy_path(
    body_template, audio_path, mock_engine, tmp_path, workdir
):
    """Synthesised body + mock MuseTalk + compositor + mux = final mp4 + metrics."""
    output_path = tmp_path / "final.mp4"
    source_image = _write_source_png(tmp_path)

    result = render_cached_avatar(
        avatar_id="avatar_001",
        gesture_id="explain_both",
        identity_id="id-cached-test",
        audio_path=audio_path,
        output_path=output_path,
        engine=mock_engine,
        source_image=source_image,
        pack_root=workdir / "packs",
        capture_dir=tmp_path / "captures",
        body_templates_dir=body_template,
    )

    # ── Filesystem layout ─────────────────────────────────────────────────────
    assert result.face_roi_path.is_file()
    assert result.face_lipsynced_path.is_file()
    assert result.composited_path.is_file()
    assert result.final_path is not None and result.final_path.is_file()
    # metrics.json + result.json land under capture_root/<job_id>/ where
    # the runtime mp4s also live (see render_cached_avatar._write_metrics).
    metrics_dir = output_path.parent / "captures" / result.face_roi_path.parent.name
    metrics_path = metrics_dir / "metrics.json"
    result_json = metrics_dir / "result.json"
    assert metrics_path.is_file()
    assert result_json.is_file()

    # ── Metrics schema (block-1 contract) ─────────────────────────────────────
    metrics = json.loads(metrics_path.read_text())
    assert metrics["body_cache_hit"] is True
    assert metrics["identity_cache_hit"] is False  # first compile
    assert metrics["face_region_only"] is True
    assert metrics["face_resolution"] == [256, 256]
    assert metrics["batch_size"] == 8
    assert metrics["output_seconds"] > 0.0
    assert metrics["wall_seconds"] > 0.0
    assert metrics["gpu_seconds"] > 0.0  # mock provides a deterministic estimate
    assert metrics["gpu_seconds_per_output_minute"] > 0.0

    # ── Result dataclass shape ────────────────────────────────────────────────
    assert isinstance(result, RenderCachedAvatarResult)
    assert result.body_cache_hit is True
    assert result.face_region_only is True
    assert result.face_resolution == (256, 256)
    assert result.batch_size == 8


@requires_ffmpeg
def test_render_cached_avatar_body_template_missing(tmp_path, audio_path, mock_engine):
    """Body template without the canonical 5 files raises FileNotFoundError."""
    source_image = _write_source_png(tmp_path)
    with pytest.raises(FileNotFoundError) as exc:
        render_cached_avatar(
            avatar_id="ghost",
            gesture_id="idle",
            identity_id="id-ghost",
            audio_path=audio_path,
            output_path=tmp_path / "final.mp4",
            engine=mock_engine,
            source_image=source_image,
            body_templates_dir=tmp_path,
        )
    msg = str(exc.value)
    for filename in ("body.mp4", "face_mask.mp4", "neck_mask.mp4", "metadata.json", "face_transforms.npz"):
        assert filename in msg


@requires_ffmpeg
def test_render_cached_avatar_identity_cache_hit(
    body_template, audio_path, mock_engine, tmp_path, workdir
):
    """Second call with pre-built pack short-circuits the compile step."""
    from src.storage.avatar_packs import AvatarPackRepository

    source_image = _write_source_png(tmp_path)

    # First call: compile a pack on the workdir pack root.
    repo_root = workdir / "packs"
    repo = AvatarPackRepository(root=repo_root)
    render_cached_avatar(
        avatar_id="avatar_001",
        gesture_id="explain_both",
        identity_id="id-cache-hit",
        audio_path=audio_path,
        output_path=tmp_path / "first.mp4",
        engine=mock_engine,
        source_image=source_image,
        pack_repo=repo,
        pack_root=repo_root,
        capture_dir=tmp_path / "captures",
        body_templates_dir=body_template,
    )
    # Pick the freshly-compiled pack.
    pack_files = sorted(repo_root.glob("id-*.tar"))
    assert pack_files, "AvatarCompiler should have written a pack to disk"
    pack_path = pack_files[0]

    # Second call: explicit pack path → identity_cache_hit = True.
    result = render_cached_avatar(
        avatar_id="avatar_001",
        gesture_id="explain_both",
        identity_id="id-cache-hit",
        audio_path=audio_path,
        output_path=tmp_path / "second.mp4",
        engine=mock_engine,
        pack_repo=repo,
        pack_root=repo_root,
        capture_dir=tmp_path / "captures",
        body_templates_dir=body_template,
        identity_pack_path=pack_path,
    )
    assert result.identity_cache_hit is True
    metrics_path = tmp_path / "captures" / result.face_roi_path.parent.name / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    assert metrics["identity_cache_hit"] is True


@requires_ffmpeg
def test_render_cached_avatar_batch_size_attribute_propagates(
    body_template, audio_path, mock_engine, tmp_path, workdir
):
    """``batch_size`` in metrics mirrors engine.render_batch_size."""
    mock_engine.render_batch_size = 4
    source_image = _write_source_png(tmp_path)
    result = render_cached_avatar(
        avatar_id="avatar_001",
        gesture_id="explain_both",
        identity_id="id-batch-4",
        audio_path=audio_path,
        output_path=tmp_path / "batch4.mp4",
        engine=mock_engine,
        source_image=source_image,
        pack_root=workdir / "packs",
        capture_dir=tmp_path / "captures",
        body_templates_dir=body_template,
    )
    metrics_path = tmp_path / "captures" / result.face_roi_path.parent.name / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    assert metrics["batch_size"] == 4
