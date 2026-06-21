"""Test for the MuseTalk real-mode upstream import helper (Task 4).

The helper tries to import ``musetalk`` from ``PYTHONPATH`` first, then
falls back to the path advertised via ``HEYAVATAR_MUSETALK_SRC``.

Doesn't require CUDA or downloading any weights — we fabricate a tiny
Python package on ``tmp_path`` with the expected shape and verify the
helper resolves it. This catches a regression where the upstream
detection contract silently breaks.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from providers.liveportrait.checkpoint_manager.manifest import CheckpointEntry
from providers.musetalk.adapter import checkpoints as muse_ckpt
from providers.musetalk.adapter._upstream import _import_musetalk_upstream


@pytest.fixture
def fabricated_musetalk(tmp_path, monkeypatch):
    """Create a fake upstream package with a top-level ``musetalk`` module."""
    root = tmp_path / "FakeMuseTalk"
    root.mkdir()
    (root / "__init__.py").write_text("", encoding="utf-8")
    (root / "musetalk.py").write_text(
        "MS_VERSION = 'fake-0.0.1'\n", encoding="utf-8"
    )
    sys.modules.pop("musetalk", None)
    monkeypatch.setenv("HEYAVATAR_MUSETALK_SRC", str(root))
    monkeypatch.syspath_prepend(str(root))
    return root


def test_musetalk_real_mode_imports_when_path_set(fabricated_musetalk):
    module = _import_musetalk_upstream()
    assert module is not None
    assert getattr(module, "MS_VERSION", None) == "fake-0.0.1"


def test_musetalk_real_mode_returns_none_when_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "HEYAVATAR_MUSETALK_SRC", str(tmp_path / "does-not-exist")
    )
    sys.modules.pop("musetalk", None)
    monkeypatch.syspath_prepend(str(tmp_path))
    assert _import_musetalk_upstream() is None


def _make_manager_with_root(tmp_path: Path, monkeypatch) -> muse_ckpt.MuseTalkCheckpointManager:
    """Build a MuseTalkCheckpointManager pointing at a tmp root directory.

    Real :class:`CheckpointEntry` (one per MuseTalk manifesto slot),
    not a stub. The hash pin can be set per test.
    """
    root = tmp_path / "musetalk_ckpt"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HEYAVATAR_MUSETALK_CHECKPOINTS", str(root))
    mgr = muse_ckpt.MuseTalkCheckpointManager()
    mgr.root = root
    mgr.entries = [
        CheckpointEntry(
            name="musetalk_unet.pth",
            url="https://example.invalid/musetalk_unet.pth",
            expected_sha256="TBD",
            expected_size_bytes=0,
        ),
    ]
    return mgr


def test_musetalk_verify_policy_mock_mode_shortcircuits(tmp_path, monkeypatch):
    """Branch 1: ``mock_mode`` short-circuits every entry to True."""
    mgr = _make_manager_with_root(tmp_path, monkeypatch)
    mgr.mock_mode = True
    assert mgr.verify() == [True]
    assert mgr.entries[0].verified is True


def test_musetalk_verify_policy_tbd_with_skip_env_accepts(tmp_path, monkeypatch):
    """Branch 2: ``TBD`` + ``HEYAVATAR_SKIP_SHA256_VERIFY=1`` → accepted."""
    monkeypatch.setenv("HEYAVATAR_SKIP_SHA256_VERIFY", "1")
    mgr = _make_manager_with_root(tmp_path, monkeypatch)
    mgr.mock_mode = False
    # Place a real (placeholder) file in the cache root.
    placeholder = mgr.root / mgr.entries[0].name
    placeholder.write_bytes(b"placeholder-bytes")
    try:
        assert mgr.verify() == [True]
        assert mgr.entries[0].verified is True
    finally:
        monkeypatch.delenv("HEYAVATAR_SKIP_SHA256_VERIFY", raising=False)
        placeholder.unlink(missing_ok=True)


def test_musetalk_verify_policy_tbd_without_skip_env_rejects(tmp_path, monkeypatch):
    """Branch 3: ``TBD`` + no skip env → refuses verification."""
    monkeypatch.delenv("HEYAVATAR_SKIP_SHA256_VERIFY", raising=False)
    mgr = _make_manager_with_root(tmp_path, monkeypatch)
    mgr.mock_mode = False
    placeholder = mgr.root / mgr.entries[0].name
    placeholder.write_bytes(b"placeholder-bytes")
    try:
        assert mgr.verify() == [False]
        assert mgr.entries[0].verified is False
    finally:
        placeholder.unlink(missing_ok=True)


def test_musetalk_verify_policy_pinned_hash_matches(tmp_path, monkeypatch):
    """Branch 4: pinned SHA matches the cached file → verified."""
    import hashlib
    monkeypatch.delenv("HEYAVATAR_SKIP_SHA256_VERIFY", raising=False)
    mgr = _make_manager_with_root(tmp_path, monkeypatch)
    mgr.mock_mode = False
    payload = b"real-bytes-for-hash"
    placeholder = mgr.root / mgr.entries[0].name
    placeholder.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    mgr.entries[0].expected_sha256 = expected
    try:
        assert mgr.verify() == [True]
        assert mgr.entries[0].verified is True
    finally:
        placeholder.unlink(missing_ok=True)


def test_musetalk_verify_policy_pinned_hash_mismatches(tmp_path, monkeypatch):
    """Branch 5: pinned SHA does NOT match the cached file → rejected."""
    monkeypatch.delenv("HEYAVATAR_SKIP_SHA256_VERIFY", raising=False)
    mgr = _make_manager_with_root(tmp_path, monkeypatch)
    mgr.mock_mode = False
    placeholder = mgr.root / mgr.entries[0].name
    placeholder.write_bytes(b"irrelevant-bytes")
    mgr.entries[0].expected_sha256 = "0" * 64
    try:
        assert mgr.verify() == [False]
        assert mgr.entries[0].verified is False
    finally:
        placeholder.unlink(missing_ok=True)

