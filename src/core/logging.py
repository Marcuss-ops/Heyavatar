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


__all__ = ["configure_logging"]
