"""CheckpointManager tests.

These tests cover (1) mock-mode is a no-op and exposes a manifest,
(2) missing pins refuse to verify, (3) calling ``ensure_present`` in
mock mode never touches the filesystem root for network resources.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from providers.liveportrait.checkpoint_manager import (
    CHECKPOINT_MANIFEST,
    CheckpointManager,
)


def test_mock_mode_manifest_is_exposed() -> None:
    # HEYAVATAR_MOCK_ENGINE=1 is set by tests/conftest.py autouse fixture.
    cm = CheckpointManager(root=Path("./checkpoints/liveportrait"))
    assert cm.mock_mode
    # The manifest lists the canonical upstream filenames.
    names = {e["name"] for e in CHECKPOINT_MANIFEST}
    assert "appearance_feature_extractor.pth" in names
    assert "motion_extractor.pth" in names


def test_mock_mode_ensure_present_no_op(tmp_path: Path) -> None:
    cm = CheckpointManager(root=tmp_path / "never-created")
    cm.ensure_present()
    # No directory created when in mock mode because the worker image
    # does not need the real weights.
    assert not (tmp_path / "never-created").exists()


def test_missing_sha_pin_fails_verification(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("HEYAVATAR_MOCK_ENGINE", raising=False)
    cm = CheckpointManager(root=tmp_path, allow_network=False)
    entry = cm.entries[0]
    # Drop a fake file with the right name; the SHA is "TBD" so verification
    # must refuse to mark it verified.
    target = cm.local_path_for(entry.name)
    target.write_bytes(b"\x00" * 32)
    assert not cm.verify(entry)
    assert not entry.verified
