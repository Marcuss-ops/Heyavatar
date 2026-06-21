"""W3C cross-process trace context propagation.

The Heyavatar engine is multi-process: FastAPI publishes jobs into a
queue (in-memory or Redis Streams) and workers consume them. Native
HTTP ``traceparent`` headers cannot survive because the worker
process never sees the original HTTP request. This module encodes
the W3C Trace Context fields (``traceparent``, optionally
``tracestate`` plus ``baggage`` if present) into the queue payload
so a child's span is still linkable to the parent.

The payload format mirrors W3C Trace Context §3.2 verbatim — we
embed the traceparent string in a single payload key
``otel_context`` (a colon-separated versioned key list). For Redis
Streams we encourage consumers to also set the same key as a
top-level stream field.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


OTEL_PAYLOAD_KEY = "otel_context"
# W3C fields we propagate. tracestate / baggage only when present.
_TRACESTATE_KEY = "tracestate"
_BAGGAGE_KEY = "baggage"

#: Where a W3C traceparent is stored inside an HTTP header.
W3C_HEADER = "traceparent"


def inject_traceparent(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Inject the current OTel context into a queue ``payload`` dict.

    Returns a shallow copy so existing callers that pass a frozen
    Pydantic model can still mutate downstream. No-op if the OTel
    SDK is not installed (e.g. lightweight worker builds).
    """
    try:
        from opentelemetry import trace, context as otel_context
        from opentelemetry.propagate import inject
    except ImportError:
        return payload

    if not payload:
        payload = {}
    headers: Dict[str, str] = {}
    inject(headers)
    if not headers:
        return payload
    encoded = _encode({k: v for k, v in headers.items()
                       if k in (W3C_HEADER, _TRACESTATE_KEY, _BAGGAGE_KEY)})
    if encoded:
        # mutate shallow copy returned to caller
        if isinstance(payload, dict):
            payload[OTEL_PAYLOAD_KEY] = encoded
        else:
            # Pydantic-like objects expose ``dict()`` and ``__setitem__``
            try:
                payload[OTEL_PAYLOAD_KEY] = encoded
            except (TypeError, AttributeError, ValueError):
                pass
    return payload


def extract_traceparent(payload: Optional[Dict[str, Any]]) -> Any:
    """Return the OTel ``Context`` extracted from a queue ``payload``.

    Returns ``None`` (or the empty context) when there is no
    propagation header. The OTel SDK is imported lazily so a
    non-instrumented worker can call this safely.
    """
    if not payload:
        return None
    encoded = payload.get(OTEL_PAYLOAD_KEY)
    if not encoded:
        return None
    decoded = _decode(encoded)
    try:
        from opentelemetry.propagate import extract
        return extract(decoded)
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def _encode(headers: Dict[str, str]) -> str:
    """Encode a small dict of W3C headers as a single carriage-return-
    delimited string. Compact, ASCII, safe to live inside JSON.
    """
    return "\r\n".join(f"{k}:{v}" for k, v in headers.items())


def _decode(blob: str) -> Dict[str, str]:
    parts: Iterable[str] = blob.split("\r\n")
    out: Dict[str, str] = {}
    for entry in parts:
        if not entry:
            continue
        if ":" not in entry:
            continue
        k, _, v = entry.partition(":")
        out[k.strip().lower()] = v.strip()
    return out


__all__ = [
    "OTEL_PAYLOAD_KEY",
    "W3C_HEADER",
    "extract_traceparent",
    "inject_traceparent",
]
