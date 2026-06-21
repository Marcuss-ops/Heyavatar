"""GPU worker entrypoint.

A persistent process that loads exactly one :class:`AvatarEngine` instance
and serves a stream of jobs from the configured JobQueue. Each render job
produces chunk videos and a manifest; the :class:`EncodingWorker` is then
invoked to trim overlap, concatenate, mux audio, and produce the final mp4.

The worker process owns the GPU exclusively. The FastAPI gateway never
imports torch, never allocates VRAM, never blocks the network.

Submodules
----------
* :mod:`worker` — the :class:`GpuWorker` dataclass (fields + lifecycle
  methods ``run`` / ``stop`` / ``_update_job_state``).
* :mod:`process` — the ``_process`` / ``_do_process`` methods attached
  to :class:`GpuWorker`. This split isolates the long job-processing
  pipeline from the lightweight run loop.
* :mod:`telemetry` — Prometheus exposition, OpenTelemetry trace
  context extraction, and the small helper utilities (``_bump_inflight``,
  ``_id_from_str``, ``read_pack_from_archive``).
* :mod:`cli` — argparse runtime entry point and queue / repo
  factory bootstrap.
"""
