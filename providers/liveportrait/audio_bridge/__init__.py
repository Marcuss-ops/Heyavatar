"""Audio-to-expression bridge for LivePortrait.

LivePortrait is an expression-driven portrait animator: its
``warp_decode`` consumes a per-frame ``kp_d[1, 21, 3]`` driving
keypoint tensor, not raw audio. The :class:`RenderChunkRequest`
contract, however, hands every adapter an ``audio_window`` and
``audio_path``. This package is the bridge in between.

Canonical entry point
---------------------
:func:`providers.liveportrait.audio_bridge.bridge.audio_to_driving` —
single function that maps an audio window to a
:class:`DrivingSignals` ready for ``warp_decode``.

Backends
--------
The :func:`audio_to_driving` dispatch is driven by
:setting:`Settings.audio_bridge_backend` (env var
``HEYAVATAR_AUDIO_BRIDGE_BACKEND``):

* ``dsp`` (default, CI, production) — pure-Python envelope mapping.
  No ML deps. Same deterministic output the legacy tests assert
  against. **This is the actively-supported backend per Change 3 /
  ROADMAP.md §1.**
* ``neural`` (frozen) — SadTalker Audio2Motion (3DMM 50+3) →
  :mod:`providers.liveportrait.audio_bridge.projection` →
  LivePortrait driving tensor. The submodule tree is preserved on
  disk for forward compatibility but the production dispatch never
  takes this branch in the MVP. Operators flipping back to
  ``neural`` MUST reinstall the
  ``pip install -e ".[audio-bridge-neural]"`` extra; SadTalker's
  audio2motion checkpoint weights are trained on LRW + VoxCeleb
  (non-commercial research corpora) so the
  ``liveportrait-human-v1.commercial_use`` flag stays ``false``
  regardless.
  **Never** auto-falls-back to DSP; missing imports surface as
  ``EngineState.DEGRADED`` so the orchestrator routes around broken
  workers.

Submodules
----------
* :mod:`types`  — passive data structures (``DrivingSignals``,
  ``SadTalkerCoefs``) shared across the bridge.
* :mod:`dsp`    — pure-Python DSP primitives (WAV reader, resampler,
  RMS/ZCR/pitch envelopes). Backend-internal; never reaches the public
  boundary. **Production path of the MVP.**
* :mod:`projection` — 3DMM(50) → LP(21, 3) static projection with
  identity placeholder matrix (calibration follows on GPU worker).
  Frozen with the SadTalker backend.
* :mod:`sadtalker` — SadTalker Audio2Motion wrapper. Raises
  RuntimeError on missing imports instead of silent fallback.
  Frozen with the SadTalker backend.
* :mod:`bridge` — the public :func:`audio_to_driving` function.

Citation
--------
The driving-tensor shape was sourced from
https://github.com/KlingAIResearch/LivePortrait/blob/main/src/live_portrait_pipeline.py
the ``forward`` method's call into ``warping_module.warp_decode``.
"""

from providers.liveportrait.audio_bridge.types import (
    DMM_EXPRESSION_DIM,
    DMM_JAW_DIM,
    EXPRESSION_DIM,
    N_KEYPOINTS,
    DrivingSignals,
    SadTalkerCoefs,
)
from providers.liveportrait.audio_bridge.bridge import audio_to_driving

__all__ = [
    "audio_to_driving",
    "DrivingSignals",
    "SadTalkerCoefs",
    "N_KEYPOINTS",
    "EXPRESSION_DIM",
    "DMM_EXPRESSION_DIM",
    "DMM_JAW_DIM",
]
