"""OpenTelemetry tracing + W3C context propagation.

Subpackages under this one
--------------------------
* :mod:`tracing` — :func:`setup_tracing`, :func:`get_tracer`,
  :func:`shutdown_tracing`. Manage the OTLP tracer provider lifecycle
  (resource attributes, BatchSpanProcessor attach, flush on exit).
* :mod:`propagation` — :func:`inject_traceparent` /
  :func:`extract_traceparent` plus the W3C header encode/decode
  helpers. The encode format mirrors W3C Trace Context §3.2 verbatim
  so a child span started by a worker process is still linkable to the
  parent FastAPI request.

Use the specific submodule for imports, e.g.
``from src.observability.distributed.tracing import setup_tracing``.
"""
