"""BodyTemplate frozen dataclass — concrete on-disk prerecorded body clip.

Per ``docs/REPOSITORY_SLIMMING_PLAN.md`` §4 + Change 4 acceptance
(``ROADMAP.md`` §3), the speculative ``BodyAssetProvider`` ABC was
removed in Change 1. This dataclass is the canonical concrete
replacement: a single ``body_templates/<avatar_id>/<gesture_id>/``
directory holding the four canonical files plus a ``metadata.json``
that the shot-list / frame-align utilities consume.

Files referenced:

* ``body.mp4``                  — the prerecorded driving video
  frames; the liveportrait + musetalk stack reads this directly.
* ``face_mask.mp4``             — per-frame convex hull of face
  landmarks (single-channel, BGR three-channel mp4 from
  ``tools/avatar_assets/precompute_video_template.py``).
* ``neck_mask.mp4``             — per-frame polygon derived from
  jaw indices, Gaussian-blurred to feather the neck transition.
* ``face_transforms.npz``       — per-frame bbox, matrices,
  landmarks, confidence, timestamp_ms arrays consumed by the
  liveportrait adapter and the frame-align utility.
* ``metadata.json``             — ``{avatar_id, gesture_id, width,
  height, fps, total_frames, status}``.

This is a pure data class: there is no engine call, no liveportrait
import, no mediapipe import — anything that needs the body template
either reads these files directly or wraps this class around the
filesystem layout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.core.logging import get_logger
from src.domain.types import IdentityId

LOG = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class BodyTemplate:
    """One real, on-disk prerecorded body segment.

    See module docstring for the canonical 5-file layout.

    The class deliberately does no validation in its constructor —
    the loader (or the frame-align utility) verifies the files exist
    so callers can build an in-memory ``BodyTemplate`` from
    pre-trusted paths (the frame-align product, the test harness,
    the synthetic body-template fixtures) without round-tripping
    through ``__post_init__``.
    """

    body_video: Path
    face_mask: Path
    neck_mask: Path
    face_transforms: Path
    metadata: Path

    # ── metadata.json-derived accessors ───────────────────────────────────────

    def _read_metadata(self) -> dict:
        if not self.metadata.is_file():
            raise FileNotFoundError(
                f"BodyTemplate metadata.json missing at {self.metadata}"
            )
        with open(self.metadata, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def avatar_id(self) -> IdentityId:
        return IdentityId(str(self._read_metadata()["avatar_id"]))

    def gesture_id(self) -> str:
        return str(self._read_metadata()["gesture_id"])

    def fps(self) -> float:
        return float(self._read_metadata()["fps"])

    def frame_count(self) -> int:
        return int(self._read_metadata()["total_frames"])

    def width(self) -> int:
        return int(self._read_metadata()["width"])

    def height(self) -> int:
        return int(self._read_metadata()["height"])

    def status(self) -> str:
        return str(self._read_metadata().get("status", "precomputed"))


def load_body_template(
    avatar_id: str,
    gesture_id: str,
    base_dir: Path | str = "body_templates",
) -> BodyTemplate:
    """Resolve a :class:`BodyTemplate` from the on-disk body_templates tree.

    Convention (per ``tools/avatar_assets/precompute_video_template.py``,
    which writes exactly this layout):

    .. code-block:: text

        body_templates/<avatar_id>/<gesture_id>/
            body.mp4
            face_mask.mp4
            neck_mask.mp4
            face_transforms.npz
            metadata.json

    Args:
        avatar_id: top-level identity (e.g. "male_business_01").
        gesture_id: gesture within that identity
            (e.g. "explain_both", "idle").
        base_dir: parent directory holding all body templates.
            Defaults to ``"body_templates"`` (the conventional
            "./body_templates" relative to the project root).

    Raises:
        FileNotFoundError: when one of the five canonical files is
            missing at the resolved path. The exception message names
            the missing files so the operator can rull ``precompute_video_template.py``
            against the right (avatar_id, gesture_id) pair.
    """
    template_dir = Path(base_dir) / avatar_id / gesture_id
    template = BodyTemplate(
        body_video=template_dir / "body.mp4",
        face_mask=template_dir / "face_mask.mp4",
        neck_mask=template_dir / "neck_mask.mp4",
        face_transforms=template_dir / "face_transforms.npz",
        metadata=template_dir / "metadata.json",
    )
    missing = [
        str(p) for p in (
            template.body_video, template.face_mask, template.neck_mask,
            template.face_transforms, template.metadata
        ) if not p.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"BodyTemplate for avatar_id={avatar_id!r}, "
            f"gesture_id={gesture_id!r} is missing files at "
            f"{template_dir}: {missing}. Run "
            "`tools/avatar_assets/precompute_video_template.py` "
            "to materialise the template."
        )
    return template


__all__ = ["BodyTemplate", "load_body_template"]
