"""Concrete job queue implementations.

Three flavours live here as submodules:

* :mod:`memory` — :class:`InMemoryJobQueue`, single-process, used for
  tests and CI.
* :mod:`null` — :class:`NullJobQueue`, silent queue that drops every
  job; useful when you want to disable queueing entirely (e.g.
  running a worker in --once mode).
* :mod:`redis` — :class:`RedisJobQueue`, Redis-Streams-backed, the
  production default. Imported lazily so the project keeps working
  without Redis installed.

All implementations honour :class:`JobQueue` from
:mod:`contracts.job_queue`.
"""
