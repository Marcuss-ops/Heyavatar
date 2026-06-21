"""Job-metadata repository split by backend.

The FastAPI gateway and the GPU worker both need durable job state
that survives restarts and is queryable by id. v1 ships two
implementations selected by ``HEYAVATAR_JOB_REPO_BACKEND``:

* :mod:`memory` — :class:`InMemoryJobRepository`, thread-safe in-process.
* :mod:`redis` — :class:`RedisJobRepository`, cross-process via Redis.

Subpackages under this one
--------------------------
* :mod:`memory` — :class:`InMemoryJobRepository`.
* :mod:`redis` — :class:`RedisJobRepository`.

Use the specific submodule for imports, e.g.
``from src.storage.jobs.memory import InMemoryJobRepository``.
"""
