"""Typed mirror of upstream LivePortrait configuration schemas.

We define typed dataclasses that mirror the upstream
``src.config.inference_config.InferenceConfig`` and
``src.config.crop_config.CropConfig`` from the official repo
https://github.com/KlingAIResearch/LivePortrait. The adapter never imports
upstream dataclasses directly at module-load time because that would
force a torch dependency on every test run; instead the adapter calls
into the upstream package lazily inside :meth:`LivePortraitAdapter.load`.

What lives here
---------------
* :class:`InferenceConfig` — engine knobs the worker usually wants to override
  (``flag_use_half_precision``, ``flag_do_torch_compile``, ``source_division``,
  ``device_id``, ``mask_crop``).

* :class:`CropConfig` — face-crop geometry (used by ``prepare_source`` to
  align the source image to the 256x256 canonical grid).

* :class:`LivePortraitSchemaVersion` — versioned schema tag persisted inside
  the Avatar Pack so future model updates can preserve old packs across
  adapter upgrades (see :mod:`src.domain.avatar_pack`).

Why mirror instead of re-export upstream types
----------------------------------------------
The upstream dataclasses are mutable and import ``torch``. By mirroring
them we keep the gates of the API stable across upstream churn, allow
the test suite to stay CPU-only, and serialise the config blob into
the Avatar Pack so we can detect a drift between the pack's creation
config and the worker's current config.

Citations
---------
* Upstream ``InferenceConfig``: https://github.com/KlingAIResearch/LivePortrait/blob/main/src/config/inference_config.py
* Upstream ``CropConfig``: https://github.com/KlingAIResearch/LivePortrait/blob/main/src/config/crop_config.py
* Repo: https://github.com/KlingAIResearch/LivePortrait
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

# Pinned schema value for the Avatar Pack's ``pack_version`` field. Any
# change to the shape of the LivePortrait pack entry files (tensor dtypes,
# transforms in ``transform_matrix.bin``) requires bumping this value
# AND adding a migration branch in :meth:`LivePortraitAdapter._open_pack`.
LIVE_PORTRAIT_PACK_VERSION = "liveportrait-v1"


@dataclass(slots=True, frozen=True)
class InferenceConfig:
    """Engine knobs that mirror upstream ``InferenceConfig``.

    Attributes named here match the upstream field names 1:1; the
    adapter's :func:`_to_upstream_inference_config` converts this
    dataclass to the upstream type at load time.
    """

    # If True the wrappers run in fp16; we default to True because the
    # upstream README recommends it for RTX 30/40 series.
    flag_use_half_precision: bool = True
    # If True we eagerly compile kernels via torch.compile on first load.
    # CI environments without a working Triton turn this off at boot.
    flag_do_torch_compile: bool = False
    # GPU device index. Set to -1 to mean whatever torch.cuda.current_device().
    device_id: int = 0
    # Warp feature grid must be divisible by this number (live: 8).
    source_division: int = 8
    # Path to a 256x256 PNG mask indicating which pixels of the
    # crop to paste back. Defaults to upstream bundled asset.
    mask_crop: Optional[Path] = None
    # Free-form passthrough for forward-compat with upstream fields we
    # have not named yet. Forward-compat is the job of the upgraders
    # listed in docs/MODEL_LICENSES.md, not the adapter.
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Path is not JSON-serialisable in plain json.dumps yet pathlib is
        # isolated to a string at serialise time.
        if d["mask_crop"] is not None:
            d["mask_crop"] = str(d["mask_crop"])
        return d


@dataclass(slots=True, frozen=True)
class CropConfig:
    """Geometry that ``prepare_source`` consumes."""

    # Source dlib/InsightFace landmark map name. Valid values upstream:
    # "ffhq", "style-aligned", "none".
    landmark_type: str = "ffhq"
    # When True the source image is x-cropped around the face.
    flag_do_crop: bool = True
    # Source image size for pasteback (HxW). 512x512 matches default
    # training resolution in upstream.
    source_image_size: int = 512
    # Rotate the source so eyes are horizontal before passing into
    # the keypoint extractor. Upstream default True.
    flag_do_rot: bool = True


@dataclass(slots=True, frozen=True)
class PackSchema:
    """Versioned schema tag written to the Avatar Pack as ``pack_version``.

    See :data:`LIVE_PORTRAIT_PACK_VERSION`.
    """

    schema_version: str = LIVE_PORTRAIT_PACK_VERSION
    upstream_repo: str = "KlingAIResearch/LivePortrait"
    upstream_url: str = "https://github.com/KlingAIResearch/LivePortrait"


__all__ = [
    "CropConfig",
    "InferenceConfig",
    "LIVE_PORTRAIT_PACK_VERSION",
    "PackSchema",
]
