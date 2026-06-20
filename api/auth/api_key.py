"""API-key auth dependency."""

from __future__ import annotations

import os
import secrets

from fastapi import Header, HTTPException, status


def require_api_key(x_api_key: str | None = Header(default=None)) -> str:
    """Validate ``X-API-Key`` against the configured environment key.

    Returns the validated key on success; raises 401 if absent or wrong.
    In dev mode (``HEYAVATAR_API_KEY`` not set) the dependency is a pass-through
    so local exploration does not require a key — but a warning is emitted.
    """
    expected = os.environ.get("HEYAVATAR_API_KEY")
    if not expected:
        return x_api_key or "dev-mode"
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )
    return x_api_key
