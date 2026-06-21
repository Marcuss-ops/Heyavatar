"""FastAPI application composition root.

The API owns no GPU; it authenticates, validates, publishes jobs to
the :class:`JobQueue`, and exposes HTTP routes for avatars + jobs +
health. This package owns the application factory and its dependencies;
:class:`Worker` is a separate process that picks jobs off the queue.

Subpackages under this one
--------------------------
* :mod:`state` — :class:`AppState` dataclass + ``lifespan`` context.
* :mod:`queue_factory` — :func:`_build_queue` (queue backend selection).
* :mod:`metrics` — Prometheus gatherer + latency middleware + ``/metrics`` mount.
* :mod:`factory` — :func:`create_app` composition root + module-level ``app``.

The Uvicorn convention ``api.app:app`` (from README.md and the Dockerfile
launch command) is preserved by re-exporting ``app`` and ``create_app``
from this package — without this re-export Uvicorn's ``module:attr``
notation has nothing to resolve. Internal code imports from
:mod:`api.app.factory` directly to keep the dependency direction
explicit.
"""

from api.app.factory import app, create_app

__all__ = ["app", "create_app"]
