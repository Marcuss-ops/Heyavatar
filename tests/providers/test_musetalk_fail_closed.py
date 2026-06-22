"""Tests for the MuseTalk fail-closed contract added to ``engine.py``.

Verifies the three behaviours from the plan's Block 3:

* Mock mode (``HEYAVATAR_MOCK_ENGINE=1``) keeps returning mock assets
  so CI / contract tests stay green.
* ``DEGRADED`` state in real mode raises :class:`RuntimeError` from
  both ``prepare_identity`` and ``render_chunk`` — never substitutes
  mock assets.
* An exception inside the real ``_real_render_chunk`` propagates
  unchanged; the orchestrator must surface ``FAILED_INFERENCE`` rather
  than silently emitting a degraded red mp4.

Notes on construction
---------------------
``tests/conftest.py`` globally sets ``HEYAVATAR_MOCK_ENGINE=1``. To
exercise the real-mode fail-closed branches we therefore construct
the adapter explicitly with ``Settings(mock_engine=False)`` (a frozen
``dataclass``, so a fresh instance is the only way). The
``_load_degraded_engine`` fixture stashes a non-mock engine that
already transitioned to ``DEGRADED`` so the tests don't need to invoke
``load()`` (which would otherwise try to import torch and fail).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pytest

from contracts.avatar_engine import EngineState
from providers.musetalk.adapter.engine import MuseTalkAdapter
from src.core.config import Settings
from src.domain.types import (
    AvatarIdentityHandle,
    IdentityId,
    RenderChunkRequest,
)
from tests._fixtures import PNG_1X1 as _PNG_1x1


# ─── settings & engine fixtures ─────────────────────────────────────────────


def _source_image(tmp_path: Path) -> Path:
    img = tmp_path / "face.png"
    img.write_bytes(_PNG_1x1)
    return img


@pytest.fixture
def mock_engine() -> Iterator[MuseTalkAdapter]:
    """``HEYAVATAR_MOCK_ENGINE=1`` engine — used to confirm mock-path is preserved."""
    eng = MuseTalkAdapter()
    eng.load()
    try:
        yield eng
    finally:
        eng.unload()


@pytest.fixture
def degraded_engine() -> Iterator[MuseTalkAdapter]:
    """Real-mode engine forced into ``DEGRADED`` state without invoking ``load()``.

    We construct the adapter with a fresh ``Settings(mock_engine=False)``
    (bypasses conftest's ``HEYAVATAR_MOCK_ENGINE=1``), then patch
    ``_state`` to ``DEGRADED`` directly. ``load()`` would otherwise
    try to import torch / CUDA and transition DEGRADED for a different
    reason, masking the test invariants.
    """
    eng = MuseTalkAdapter(settings=Settings(mock_engine=False))
    eng._state = EngineState.DEGRADED
    eng._last_error = "simulated upstream import failure"
    yield eng


@pytest.fixture
def failing_real_engine() -> Iterator[MuseTalkAdapter]:
    """Real-mode engine whose ``_real_render_chunk`` always raises.

    Construction mirrors ``degraded_engine`` but state is set to
    ``IDLE`` so the call goes through the real-path branch (the only
    one that calls ``_real_render_chunk``).
    """
    eng = MuseTalkAdapter(settings=Settings(mock_engine=False))
    eng._state = EngineState.IDLE
    eng._last_error = None
    yield eng


# ─── tests ──────────────────────────────────────────────────────────────────


def test_mock_mode_returns_assets(mock_engine, tmp_path):
    """HEYAVATAR_MOCK_ENGINE=1 must keep returning the synthetic dict."""
    assets = mock_engine.prepare_identity(_source_image(tmp_path))
    assert isinstance(assets, dict)
    assert "source_latent.bin" in assets
    assert "identity_embedding.bin" in assets


def test_degraded_state_in_real_mode_raises_prepare_identity(degraded_engine, tmp_path):
    """A DEGRADED engine must raise on prepare_identity, never ship mock packs."""
    with pytest.raises(RuntimeError) as exc:
        degraded_engine.prepare_identity(_source_image(tmp_path))
    assert "DEGRADED" in str(exc.value)
    assert "simulated upstream import failure" in str(exc.value)


def test_degraded_state_in_real_mode_raises_render_chunk(degraded_engine):
    """render_chunk while DEGRADED must raise — never a fallback mp4."""
    handle = AvatarIdentityHandle(
        identity_id=IdentityId("id-x"),
        pack_path=Path("/tmp/pack.tar"),
        pack_digest="deadbeef",
        prepared_at=datetime.now(timezone.utc),
    )
    request = RenderChunkRequest(
        job_id="job-test",
        audio_window=(0.0, 1.0),
        audio_path=Path("/tmp/speech.wav"),
        fps=25,
        resolution=(512, 512),
        chunk_index=0,
        face_region_only=False,
    )
    with pytest.raises(RuntimeError) as exc:
        degraded_engine.render_chunk(request, handle)
    assert "DEGRADED" in str(exc.value)


def test_real_render_exception_propagates(failing_real_engine, monkeypatch):
    """When _real_render_chunk raises, render_chunk must NOT swallow it."""
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated upstream failure")

    monkeypatch.setattr(failing_real_engine, "_real_render_chunk", _boom)

    handle = AvatarIdentityHandle(
        identity_id=IdentityId("id-x"),
        pack_path=Path("/tmp/pack.tar"),
        pack_digest="deadbeef",
        prepared_at=datetime.now(timezone.utc),
    )
    request = RenderChunkRequest(
        job_id="job-test",
        audio_window=(0.0, 1.0),
        audio_path=Path("/tmp/speech.wav"),
        fps=25,
        resolution=(256, 256),
        chunk_index=0,
        face_region_only=True,
    )
    with pytest.raises(RuntimeError) as exc:
        failing_real_engine.render_chunk(request, handle)
    assert "simulated upstream failure" in str(exc.value)


def test_settings_is_frozen_default_no_mock():
    """Sanity: Settings(mock_engine=False) bypasses conftest's HEYAVATAR_MOCK_ENGINE=1."""
    settings = Settings(mock_engine=False)
    assert settings.mock_engine is False
    with pytest.raises((AttributeError, Exception)):
        settings.mock_engine = True  # type: ignore[misc]
