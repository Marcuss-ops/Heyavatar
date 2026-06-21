"""Prometheus metrics facade for the Heyavatar engine.

This package registers a small, *low-cardinality* metric set on the
default Prometheus collector registry. The headline economic metric
— **GPU-seconds per minute of useful output** — is therefore
expressed as TWO counters that operators combine in PromQL with
:func:`rate`. This is the only way to get a correct ratio in
multi-process Prometheus where each worker publishes its own
cumulative totals.

Submodules
----------
* :mod:`constants` — shared low-cardinality label sets.
* :mod:`instruments` — declarations of the Prometheus ``Counter``,
  ``Gauge``, and ``Histogram`` metric objects bound to the default
  collector registry.
* :mod:`recorders` — typed wrappers the application code calls in
  preference to touching the instruments directly. This is the only
  public surface most modules should import.
* :mod:`exposition` — helpers for emitting the metrics surface to
  Prometheus (used by the FastAPI ``/metrics`` route and the
  standalone worker HTTP server).

Why Counters, not a Gauge
--------------------------
A precomputed rolling-window ``Gauge`` updated by a Python
background thread is a known anti-pattern:

* It loses data when a worker dies.
* Cross-worker aggregation of precomputed rolling ratios is the
  average-of-averages fallacy.
* It cannot be slice-and-diced in Grafana ("ratio in the last
  30 seconds vs last hour" is impossible without recomputing in
  Prometheus).

Two Counters (``gpu_seconds_total``, ``output_minutes_total`` plus
``rate()`` in the dashboard give us:

* Crash-safe cumulative totals.
* Cross-worker summation that is mathematically correct.
* Arbitrary windowing in Grafana.

Metric cardinality contract
---------------------------
Labels in this package are **strictly** limited to low-cardinality
``engine_id`` (``"musetalk-v1"`` … ``"liveportrait-human-v1"``)
and ``tier`` (``"express"`` / ``"studio"`` / ``"premium"``).
**Never** add ``job_id`` / ``identity_id`` / ``worker_id`` /
``request_path`` to any of these metrics. The historian sidetrack
in ``docs/observability.md`` includes a CIDR-block scraper
configuration that omits high-cardinality labels.
"""
