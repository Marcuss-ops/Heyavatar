"""Logging configuration backed by Loguru."""

from __future__ import annotations

import sys

from loguru import logger as _loguru

from .config import Settings, get_settings


def configure_logging(settings: Settings | None = None) -> None:
    """Initialise Loguru with a single stderr sink and ISO timestamp."""
    settings = settings or get_settings()
    _loguru.remove()
    if settings.log_json:
        _loguru.add(
            sys.stderr,
            level=settings.log_level,
            serialize=True,
            backtrace=False,
            diagnose=False,
        )
    else:
        _loguru.add(
            sys.stderr,
            level=settings.log_level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> "
                "<level>[{level:>5}]</level> {name}: {message}"
            ),
            backtrace=False,
            diagnose=False,
        )


def get_logger(name: str | None = None):
    """Return a per-module Loguru logger.

    Mirrors the stdlib ``logging.getLogger`` API so module-level
    ``LOG = get_logger(__name__)`` keeps working across the project
    without the import dance. ``name`` is forwarded as the loguru
    ``name`` extra so messages can be filtered by module.
    """
    return _loguru.bind(name=name) if name else _loguru


__all__ = ["configure_logging", "get_logger"]



