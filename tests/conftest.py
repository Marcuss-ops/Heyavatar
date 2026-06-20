"""Pytest fixtures for the Heyavatar engine.

Toggles the mock engine globally so every adapter under test runs on CPU
without requiring CUDA. Also exposes temp directories so tests don't
write into the project root.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pytest


@pytest.fixture(scope="session", autouse=True)
def _force_mock_engine() -> None:
    os.environ["HEYAVATAR_MOCK_ENGINE"] = "1"
    # Clear cached settings so the new env var takes effect.
    from src.core.config import get_settings
    get_settings.cache_clear()


@pytest.fixture
def workdir(tmp_path: Path) -> Iterator[Path]:
    """A temporary workdir; sets HEYAVATAR_PACK_DIR etc. for the test."""
    os.environ["HEYAVATAR_PACK_DIR"] = str(tmp_path / "packs")
    os.environ["HEYAVATAR_CAPTURE_DIR"] = str(tmp_path / "captures")
    os.environ["HEYAVATAR_OBJECT_STORE"] = str(tmp_path / "object_store")
    os.environ["HEYAVATAR_REGISTRY"] = "registry/models.yaml"
    # Clear cached settings so env overrides take effect.
    from src.core.config import get_settings
    get_settings.cache_clear()
    (tmp_path / "packs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "captures").mkdir(parents=True, exist_ok=True)
    yield tmp_path
    get_settings.cache_clear()
