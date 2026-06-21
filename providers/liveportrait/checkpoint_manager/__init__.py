"""Checkpoint manager for LivePortrait model weights.

The official weights are hosted on HuggingFace Hub and downloaded via
``huggingface_hub`` Python API (preferred, handles LFS + progress bar)
or plain ``urllib`` as fallback. Files are cached to
``$HEYAVATAR_LIVE_PORTRAIT_CHECKPOINTS`` (default
``./checkpoints/liveportrait/``).

Why this is its own module
--------------------------
* The downstream worker process is the only place that should ever
  perform a ~2 GB download, and we want to time that work outside the
  hot render path.
* Verification MUST happen before the weights are loaded into VRAM;
  otherwise a corrupted file would crash mid-render and corrupt the
  in-flight job.
* Tests must never touch the network. The ``__init__`` short-circuits
  on ``HEYAVATAR_MOCK_ENGINE=1`` and only exposes the manifest.

Submodules
----------
* :mod:`manifest` — pins + lightweight ``CheckpointEntry`` dataclass.
* :mod:`downloader` — HF Hub + urllib download primitives and the
  ``_download_to_smart`` selector.
* :mod:`manager` — the ``CheckpointManager`` orchestrator that combines
  discovery, verification, and the high-level ``ensure_present`` call.

Cisco file layout produced
--------------------------

::

    <root>/
        appearance_feature_extractor.pth
        motion_extractor.pth
        warping_module.pth
        stitching_retargeting_module.pth
        spade_generator.pth
        manifest.json     # {"version": "...", "files": [{"name":..., "sha256":..., "size_bytes":...}, ...]}

Citations
---------

* LivePortrait upstream: https://github.com/KlingAIResearch/LivePortrait
* HuggingFace Hub: https://huggingface.co/KlingTeam/LivePortrait
"""
