"""Encoding worker — assembles rendered chunks into the final video.

Reads a chunk manifest produced by :class:`RenderVideo`, trims the
overlap between consecutive chunks, concatenates them, muxes the
original audio, and emits the final mp4 (or webm).

The overlap trimming solves the problem where chunk N and chunk N+1
share ``overlap_seconds`` of context that would otherwise appear twice
in the output. Each chunk is trimmed to its non-overlapping segment
before concatenation.

Subpackages under this one
---------------------------
* :mod:`worker` — :class:`EncodingWorker` and the per-chunk encode pipeline.
* :mod:`manifest` — :func:`_parse_manifest` chunk-list parser.
* :mod:`codec` — :func:`_resolve_codec` h264_NVENC/libx264 dispatch.
* :mod:`cli` — :func:`main` argparse entry point.

Use the specific submodule for imports, e.g.
``from workers.encoding_worker.worker import EncodingWorker``.
"""
