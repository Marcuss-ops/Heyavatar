"""Lazy import helpers and upstream-config translation.

* :func:`_import_torch` — only loaded in real-mode codepaths.
* :func:`_import_upstream_live_portrait` — dynamic-package loader that
  registers the upstream ``LivePortrait/src/`` directory under a
  unique name (``liveportrait_upstream``) to avoid colliding with
  our own ``src/`` package.
* :func:`_to_upstream_inference_config` — translates our
  :class:`InferenceConfig` dataclass + :class:`CheckpointManager` to
  the upstream side, pointing weights at our managed cache.
* :func:`_to_upstream_crop_config` /
  :func:`_crop_to_dict` / :func:`_get_wrapper` —
  shape adaptation for upstream's dataclasses and defensive
  attribute extraction.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from providers.liveportrait.checkpoint_manager.manager import CheckpointManager
from providers.liveportrait.inference_config import CropConfig, InferenceConfig
from src.core.logging import get_logger

LOG = get_logger("providers.liveportrait")


# Single source of truth for the dynamic upstream package name. Any code
# path that imports upstream modules (``liveportrait_upstream.utils.crop``
# in ``_render.py``, etc.) MUST use this constant — never a duplicated
# string literal — so renames stay in one place.
LIVE_PORTRAIT_UPSTREAM_PKG_NAME: str = "liveportrait_upstream"


def _import_torch() -> Any:
    try:
        import torch  # type: ignore
        return torch
    except ImportError:
        return None


def _import_upstream_live_portrait() -> Any:
    """Import the upstream ``live_portrait_pipeline`` module.

    To avoid package shadowing with our own ``src/`` directory, we dynamically
    register the upstream ``LivePortrait/src/`` directory as a unique package
    named ``liveportrait_upstream`` in ``sys.modules``.
    """
    import importlib
    import importlib.util
    import sys

    pkg_name = LIVE_PORTRAIT_UPSTREAM_PKG_NAME
    module_name = f"{pkg_name}.live_portrait_pipeline"

    # If already imported, just return it
    if module_name in sys.modules:
        return sys.modules[module_name]

    extra = os.environ.get("HEYAVATAR_LIVE_PORTRAIT_SRC")
    if not extra:
        extra = "./LivePortrait"

    src_path = Path(extra).resolve()
    if src_path.name != "src" and (src_path / "src").is_dir():
        src_path = src_path / "src"

    if not src_path.is_dir():
        LOG.warning("Upstream src directory not found at: %s", src_path)
        return None

    # Ensure __init__.py exists in the upstream src directory
    init_path = src_path / "__init__.py"
    if not init_path.exists():
        try:
            init_path.touch()
        except Exception as e:
            LOG.warning("Could not create __init__.py in %s: %s", src_path, e)

    try:
        spec = importlib.util.spec_from_file_location(
            pkg_name,
            str(init_path),
            submodule_search_locations=[str(src_path)]
        )
        if spec is None or spec.loader is None:
            LOG.warning("Failed to create spec for dynamic package %s", pkg_name)
            return None

        pkg = importlib.util.module_from_spec(spec)
        sys.modules[pkg_name] = pkg
        spec.loader.exec_module(pkg)

        return importlib.import_module(module_name)
    except Exception as exc:
        LOG.warning("Failed to import from dynamic package: %s", exc)
        return None


def _to_upstream_inference_config(
    upstream: Any,
    local: InferenceConfig,
    checkpoints: CheckpointManager,
) -> Any:
    """Translate our :class:`InferenceConfig` to the upstream dataclass.

    Falls back to upstream's default constructor when the upstream
    class isn't importable; the *required* fields are forwarded and
    ``extra`` is deep-copied.

    Checkpoint paths are resolved from ``checkpoints`` so the upstream
    code always loads weights from our managed cache rather than its
    own ``pretrained_weights/`` defaults.
    """
    try:
        cls = getattr(upstream, "InferenceConfig", None)
        if cls is None:
            return local.to_dict()
        return cls(
            flag_use_half_precision=local.flag_use_half_precision,
            flag_do_torch_compile=local.flag_do_torch_compile,
            device_id=local.device_id,
            source_division=local.source_division,
            mask_crop=str(local.mask_crop) if local.mask_crop else None,
            # Point the upstream wrapper at our managed checkpoint cache
            # so it never depends on the upstream pretrained_weights/ layout.
            checkpoint_F=str(checkpoints.local_path_for("appearance_feature_extractor.pth").resolve()),
            checkpoint_M=str(checkpoints.local_path_for("motion_extractor.pth").resolve()),
            checkpoint_G=str(checkpoints.local_path_for("spade_generator.pth").resolve()),
            checkpoint_W=str(checkpoints.local_path_for("warping_module.pth").resolve()),
            checkpoint_S=str(checkpoints.local_path_for("stitching_retargeting_module.pth").resolve()),
        )
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Could not build upstream InferenceConfig, falling back to dict: %s", exc)
        return local.to_dict()


def _to_upstream_crop_config(upstream: Any, local: CropConfig) -> Any:
    """Mirror of :func:`_to_upstream_inference_config` for ``CropConfig``."""
    try:
        cls = getattr(upstream, "CropConfig", None)
        if cls is None:
            return _crop_to_dict(local)
        return cls(
            landmark_type=local.landmark_type,
            flag_do_crop=local.flag_do_crop,
            source_image_size=local.source_image_size,
            flag_do_rot=local.flag_do_rot,
        )
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Could not build upstream CropConfig, falling back to dict: %s", exc)
        return _crop_to_dict(local)


def _crop_to_dict(local: CropConfig) -> Dict[str, Any]:
    """Serialise the :class:`CropConfig` to a plain dict for the upstream fallback."""
    return {
        "landmark_type": local.landmark_type,
        "flag_do_crop": local.flag_do_crop,
        "source_image_size": local.source_image_size,
        "flag_do_rot": local.flag_do_rot,
    }


def _get_wrapper(pipeline: Any) -> Any:
    """Pull ``pipeline.live_portrait_wrapper`` defensively."""
    wrapper = getattr(pipeline, "live_portrait_wrapper", None)
    if wrapper is None:
        # Older/newer upstream spellings: try attribute aliases.
        for alt in ("wrapper", "_wrapper"):
            wrapper = getattr(pipeline, alt, None)
            if wrapper is not None:
                break
    if wrapper is None:
        raise RuntimeError(
            "Upstream LivePortraitPipeline exposes no wrapper attribute; "
            "check the upstream version pinned in checkpoint_manager.manager."
        )
    return wrapper
