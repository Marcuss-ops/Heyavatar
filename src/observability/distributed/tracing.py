"""OpenTelemetry tracer setup.

Lazily imports the OpenTelemetry SDK and OTLP exporter so the rest
of the codebase can import ``tracing.py`` even when
``prometheus-client`` / ``opentelemetry-*`` are not installed.
``setup_tracing(settings)`` is a no-op if the OTLP endpoint is not
configured.

**FROZEN per Change 3 / ROADMAP.md §1.** Heyavatar's MVP relies on
structured JSON logs with ``job_id``, ``avatar_id``, ``engine id``,
``stage duration``, ``GPU seconds``, ``terminal state``, and
``failure reason`` (see ``src/core/logging.py``). OTLP exporters,
Grafana dashboards, W3C propagation at scale, and an OTLP collector
deployment are explicitly *out* of the active runtime. The
:class:`Settings.otel_endpoint` field is preserved for forward
compatibility only; ``setup_tracing`` always
short-circuits unless ``OTEL_EXPORTER_OTLP_ENDPOINT`` is explicitly
set.

Re-introduction gate (``docs/REPOSITORY_SLIMMING_PLAN.md`` §5):
an OTLP collector wired by ops AND a measurement showing traces
provide operational value beyond the structured-log baseline. Until
then this module's job is to *not break* the rest of the codebase
when the SDK is not installed.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from src.core.config import Settings
from src.core.logging import get_logger


LOG = get_logger(__name__)

_SERVICE_NAME = "heyavatar"


_initialised = False
_provider: Optional[Any] = None


def setup_tracing(settings: Settings) -> None:
    """Initialise the global OpenTelemetry tracer provider.

    No-op if ``settings.otel_endpoint`` is empty or starts with
    ``"off"``. Idempotent: a second call is silently ignored so
    FastAPI ``lifespan`` and worker ``main()`` can both call it.
    """
    global _initialised, _provider

    if _initialised:
        return

    endpoint = (settings.otel_endpoint or "").strip()
    if not endpoint or endpoint.lower().startswith("off"):
        LOG.info("OpenTelemetry disabled (otel_endpoint=%r).", settings.otel_endpoint)
        _initialised = True  # so we don't try again
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        LOG.warning(
            "OTLP endpoint set but OpenTelemetry SDK import failed; tracing disabled. "
            "Install with `pip install -e \".[observability]\"`. Underlying: %s",
            exc,
        )
        _initialised = True
        return

    resource = Resource.create({
        "service.name": _SERVICE_NAME,
        "service.version": "0.2.0",
        "deployment.environment": os.environ.get("HEYAVATAR_ENV", "dev"),
        "process.role": os.environ.get(
            "HEYAVATAR_PROCESS_ROLE", "unspecified"
        ),
    })
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
    )
    trace.set_tracer_provider(provider)
    _provider = provider
    _initialised = True
    LOG.info("OpenTelemetry tracing initialised; endpoint=%s", endpoint)


def get_tracer(name: str = "heyavatar") -> Any:
    """Return the ``Heyavatar`` tracer; safe to call before init."""
    from opentelemetry import trace
    return trace.get_tracer(name)


def shutdown_tracing() -> None:
    """Flush and shut down the global provider. Call before process exit."""
    global _initialised, _provider
    if _provider is None:
        _initialised = False
        return
    try:
        _provider.shutdown()
    finally:
        _provider = None
        _initialised = False


__all__ = ["get_tracer", "setup_tracing", "shutdown_tracing"]
