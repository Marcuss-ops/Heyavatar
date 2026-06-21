"""src.pipeline — canonical pipeline operators for Heyavatar.

Per Change 2 of `docs/REPOSITORY_SLIMMING_PLAN.md` §4 this package is
the canonical home for the production compositing implementation,
replacing the previous `providers/compositing/opencv_face/` location.
Both the runtime path (the GPU worker invoking the compositor via the
``contracts.compositor.Compositor`` ABC) and the offline preview tool
(``tools/avatar_assets/preview_face_composite.py`` /
``render_clean_composite.py``) use the same class.

The package houses the public surface of the compositor contract:
- :func:`OpenCVFaceCompositor` — the only concrete compositor for MVP.
- :func:`match_mean_std` — the colour-matching helper re-used by
  tests, kept public so future in-process tools can share it.

Adding more concrete compositors would introduce an ABC subclass per
implementation; for the single-MVP envelope the class itself is the
canonical entry point.
"""

from src.pipeline.compositor import OpenCVFaceCompositor, match_mean_std

__all__ = ["OpenCVFaceCompositor", "match_mean_std"]
