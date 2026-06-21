"""Chunk retry tests for :class:`RenderVideo` split by scenario.

Verifies that a failed chunk is retried up to ``chunk_retry_max`` times,
that exhausted retries produce a degraded fallback, and that the job
completes with a mix of successful and degraded chunks.

Scenario files
--------------
* :mod:`test_retry_succeeds` — engine fails some, succeeds on retry.
* :mod:`test_retry_exhausted` — engine always fails, fallback kicks in.
* :mod:`test_mixed_chunks` — partial success + partial degraded.
* :mod:`test_retry_attempts` — call count is exact.
* :mod:`test_degraded_output` — fallback mp4 is non-empty / valid.
* :mod:`test_retry_budget` — ``chunk_retry_max=1`` short-circuits.

The ``FailingEngine`` and ``SelectiveFailingEngine`` test doubles and
the ``requires_ffmpeg`` marker live in :mod:`_helpers`.
"""
