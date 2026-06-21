"""Render-video use case (split into thematic modules).

Splits the audio input into a sequence of render windows (anchored at
chunk boundaries, with optional overlap), invokes the adapter for each
window with **per-chunk retry**, publishes GPU-seconds telemetry per
chunk (so failed jobs don't lose accounting), and writes a manifest for
the :class:`EncodingWorker` to assemble the final video.

Submodules
----------
* :mod:`config` — :class:`ChunkConfig` policy object (chunk size,
  overlap, retry budget).
* :mod:`audio_probe` — ``ffprobe`` subprocess wrapper for audio
  duration discovery.
* :mod:`manifest` — chunk-list manifest writer consumed by the
  :class:`EncodingWorker`.
* :mod:`use_case` — :class:`RenderVideo` orchestrator that drives
  the engine through chunked rendering with retry + telemetry +
  degraded-fallback.
"""
