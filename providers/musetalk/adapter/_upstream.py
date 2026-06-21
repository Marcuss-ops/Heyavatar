"""Lazy import helpers for the MuseTalk adapter.

Only used from real-mode codepaths. Mock-mode never imports torch
or upstream MuseTalk so the CPU/CI pipeline stays import-safe.
"""

from __future__ import annotations

import os
from typing import Any

from src.core.logging import get_logger


def _import_torch() -> Any:
    try:
        import torch  # type: ignore
        return torch
    except ImportError:
        return None


def _import_musetalk_upstream() -> Any:
    """Import the upstream MuseTalk package.

    Tries two locations:

    1. ``musetalk`` (if on ``PYTHONPATH``).
    2. ``HEYAVATAR_MUSETALK_SRC`` env var appended to ``sys.path``.
    """
    import importlib
    import sys

    try:
        return importlib.import_module("musetalk")
    except ImportError:
        pass
    extra = os.environ.get("HEYAVATAR_MUSETALK_SRC")
    if extra:
        sys.path.insert(0, extra)
        try:
            return importlib.import_module("musetalk")
        except ImportError as exc:
            get_logger(__name__).warning(
                "HEYAVATAR_MUSETALK_SRC=%s did not expose musetalk: %s",
                extra,
                exc,
            )
    return None
