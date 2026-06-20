"""Shared test fixtures for the Heyavatar engine.

Provides a known-valid 1×1 PNG (``PNG_1X1``) and a temp workdir
configured for mock mode. Tests import ``PNG_1X1`` instead of inline
hex strings so a typo doesn't break collection.
"""

from __future__ import annotations

from pathlib import Path

import pytest


PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a"
    "0000000d49484452"
    "0000000100000001"
    "08060000001f15c4"
    "89"
    "0000000049454e44"
    "ae426082"
)


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    """Configure HEYAVATAR_* env vars and return a sandboxed temp dir."""
    import os
    os.environ["HEYAVATAR_MOCK_ENGINE"] = "1"
    os.environ["HEYAVATAR_PACK_DIR"] = str(tmp_path / "packs")
    os.environ["HEYAVATAR_CAPTURE_DIR"] = str(tmp_path / "captures")
    os.environ["HEYAVATAR_OBJECT_STORE"] = str(tmp_path / "object_store")
    os.environ["HEYAVATAR_QUEUE_BACKEND"] = "memory"
    os.environ["HEYAVATAR_REGISTRY"] = "registry/models.yaml"
    (tmp_path / "packs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "captures").mkdir(parents=True, exist_ok=True)
    yield tmp_path
