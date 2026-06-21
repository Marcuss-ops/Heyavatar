"""Engine-selection layer — tier routing + worker pool.

Subpackages under this one
--------------------------
* :mod:`router` — :class:`TierRouter` + :class:`RoutingDecision`. Reads
  ``registry/models.yaml`` for per-tier primary engine + fallback list,
  and optionally picks an engine from the available capacity in a
  :class:`WorkerPool`.
* :mod:`worker_pool` — :class:`WorkerPool` + :class:`WorkerRecord`.
  Thread-safe live-worker registry; tracks heartbeats, VRAM, and
  in-flight job count per worker process.

Use the specific submodule for imports, e.g.
``from src.scheduler.routing.router import TierRouter``.
"""
