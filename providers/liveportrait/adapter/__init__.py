"""LivePortrait adapter — implements :class:`AvatarEngine` for LivePortrait.

Source of truth: https://github.com/KlingAIResearch/LivePortrait

This package wires our :class:`contracts.avatar_engine.AvatarEngine`
contract to the upstream LivePortrait repo. The upstream package is
**not** a pip dependency — production deployments clone the repo into
the worker image, install its ``requirements.txt``, build the
custom CUDA op (``MultiScaleDeformableAttention``) once via
``tools/prepare_env.sh``, and add the upstream ``src/`` directory to
``PYTHONPATH``.

Upstream entry points called
----------------------------
* ``live_portrait_pipeline.LivePortraitPipeline(inf_cfg, crop_cfg)`` —
  constructor; inits the wrappers.
* ``pipeline.live_portrait_wrapper.appearance_feature_extractor`` —
  ``extract_feature_3d(img_tensor) -> f_s``.
* ``pipeline.live_portrait_wrapper.motion_extractor`` —
  ``get_kp_info(img_tensor) -> dict``.
* ``pipeline.live_portrait_wrapper.warping_module`` —
  ``warp_decode(f_s, kp_s, kp_d)``.
* ``pipeline.live_portrait_wrapper.stitching_retargeting_module`` —
  ``stitching(kp_s, kp_d)``.

Mode switch
-----------
The adapter toggles between **real mode** and **mock mode** using
``HEYAVATAR_MOCK_ENGINE``: when unset, the adapter attempts real
imports and lifecycle; when set (the default in CI tests), it
short-circuits and returns deterministic synthetic data so the
surrounding pipeline can be exercised end-to-end.

License and upstream attribution
--------------------------------
LivePortrait code + weights are MIT-licensed at the upstream repo.
The bundled landmark detector is **InsightFace buffalo_l** which is
non-commercial — replace with **MediaPipe Face Landmarker** for
production. See ``docs/MODEL_LICENSES.md`` for the obligation list
and the action required to flip ``commercial_use`` to true.

Submodules
----------
* :mod:`_mock` — deterministic mock-mode helpers.
* :mod:`_upstream` — lazy package importer + upstream-config
  translators that map our :class:`InferenceConfig` / :class:`CropConfig`
  dataclasses to the upstream side.
* :mod:`_identity` — real-mode identity preparation (face-crop
  serialisation, mask rendering, source-feature pooling).
* :mod:`_render` — real-mode per-chunk rendering (batch source
  loading, driving keypoint construction, batched
  ``warp_decode``).
* :mod:`engine` — :class:`LivePortraitAdapter` dataclass and its
  lifecycle.
"""
