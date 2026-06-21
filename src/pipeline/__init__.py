"""src.pipeline — canonical pipeline operators for Heyavatar.

Per Change 2 of `docs/REPOSITORY_SLIMMING_PLAN.md` §4 (extended to the
QC layer) this package is the canonical home for the production
compositing and post-production-quality primitives. It replaces the
previous locations of:

* the OpenCV face compositor (``providers/compositing/opencv_face/``,
  gone)
* the concrete quality checker (``src/quality/video_quality.py``,
  gone — :class:`src.quality.exceptions.{CompositeError,
  EncodingError, QualityError}` stays at its original package because
  it is shared across the pipeline path.)

Both the runtime path (the GPU worker / orchestrator invoking the
compositor via the ``contracts.compositor.Compositor`` ABC) and the
offline preview tool
(``tools/avatar_assets/preview_face_composite.py`` /
``render_clean_composite.py``) use the same class. The contract
interfaces (``contracts.compositor.Compositor``,
``contracts.quality_checker.QualityChecker``) and class names
(``OpenCVFaceCompositor``, ``VideoQualityChecker``) are unchanged.

The package houses the public surface of these two contracts:

Compositor:

- :class:`OpenCVFaceCompositor` — the only concrete compositor for MVP.
- :func:`match_mean_std` — the colour-matching helper re-used by
  tests, kept public so future in-process tools can share it.

Quality checker:

- :class:`VideoQualityChecker` — the only concrete QC for MVP.
- :func:`debug_green_ratio` / :func:`mean_luminance` — low-level
  per-frame helpers exposed for unit tests + tool scripts.
- :func:`probe_video_duration` / :func:`probe_audio_duration` /
  :func:`probe_video_codec` — ffprobe wrappers used internally and by
  tests that need to mock the duration / codec probes.

Adding more concrete compositors / QC providers would introduce an
ABC subclass per implementation; for the single-MVP envelope the
classes themselves are the canonical entry points.

Import convention
-----------------
- Runtime + offline-tool code reads from this package's surface,
  e.g. ``from src.pipeline import OpenCVFaceCompositor,
  VideoQualityChecker, debug_green_ratio``.
- Test files that need to ``monkeypatch.setattr("src.pipeline.quality.<func>",
  ...)`` pull the affected names directly from the submodule,
  e.g. ``from src.pipeline.quality import VideoQualityChecker``,
  so the patch target string and the import site are co-located
  and self-documenting.
- Test files that don't monkeypatch the QC helpers also use the
  package surface (``from src.pipeline import debug_green_ratio``)
  to keep the public boundary.
"""

from src.pipeline.compositor import OpenCVFaceCompositor, match_mean_std
from src.pipeline.quality import (
    VideoQualityChecker,
    debug_green_ratio,
    mean_luminance,
    probe_audio_duration,
    probe_video_codec,
    probe_video_duration,
)

__all__ = [
    # Compositor surface
    "OpenCVFaceCompositor",
    "match_mean_std",
    # Quality surface
    "VideoQualityChecker",
    "debug_green_ratio",
    "mean_luminance",
    "probe_video_duration",
    "probe_audio_duration",
    "probe_video_codec",
]
