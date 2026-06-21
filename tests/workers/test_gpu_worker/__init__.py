"""Worker state transition tests for :meth:`GpuWorker._do_process`.

Verifies that the worker returns the correct :class:`JobState` for
each terminal outcome: COMPLETED, COMPLETED_DEGRADED,
FAILED_INFERENCE, and FAILED_ENCODING.

Scenario files
--------------
* :mod:`test_compile_job` — compile-only flow returns COMPLETED.
* :mod:`test_completed` — all chunks succeed → COMPLETED + output_path.
* :mod:`test_completed_degraded` — partial degradation → COMPLETED_DEGRADED.
* :mod:`test_failed_inference` — total degradation → FAILED_INFERENCE.
* :mod:`test_failed_encoding` — encoding crash → FAILED_ENCODING.

Shared helpers (the ``requires_ffmpeg`` marker, fixture factories, and
chunks/result builders) live in :mod:`_helpers`.
"""
