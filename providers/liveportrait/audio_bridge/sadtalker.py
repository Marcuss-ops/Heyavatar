"""SadTalker Audio2Motion adapter.

Wraps the upstream SadTalker audio-to-motion subroutine to produce per-frame
3DMM expression + jaw coefficients that :mod:`projection` then folds
into LivePortrait's per-frame keypoint delta.

Production path (real GPU box)
-----------------------------
1. ``HEYAVATAR_AUDIO_BRIDGE_BACKEND=neural`` is required.
2. ``pip install -e ".[audio-bridge-neural]"`` pre-installs SadTalker.
3. The upstream package is imported here lazily so CI without CUDA
   does NOT pay the import cost on the hot path.

Failure mode (deliberate)
-------------------------
When ``HEYAVATAR_AUDIO_BRIDGE_BACKEND=neural`` is set and SadTalker
is not importable, this module **RAISES** :class:`RuntimeError`. We
do NOT silently fall back to the DSP bridge — that policy is
explicitly chosen so a worker with broken CUDA extensions cannot
silently ship DSP-tier mouth motion to paying customers. The
:func:`providers.liveportrait.audio_bridge.bridge.audio_to_driving`
caller surfaces the failure by transitioning the engine to
``EngineState.DEGRADED`` so the orchestrator routes around the
broken worker.

Mock-mode / CI path
-------------------
In :envvar:`HEYAVATAR_MOCK_ENGINE=1` mode the render loop is
short-circuited before this module is reached (see
``providers.liveportrait.adapter.engine.render_chunk``), so the
import-failure semantics above only matter for real-mode workers.
Test fixtures under ``tests/providers/`` stub the SadTalker import
via ``monkeypatch.setitem(sys.modules, "sadtalker", ...)`` so the
unit suite can exercise the wiring without GPU / SadTalker.
"""


from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from src.core.logging import get_logger


LOG = get_logger(__name__)


_SADTALKER_IMPORT_ERROR: Optional[ImportError] = None


def _try_import_sadtalker():
    """Lazy import of SadTalker's audio-to-motion submodule.

    Caches the failure so we don't pay the import cost twice. The
    cache key is the exception object identity — if a worker somehow
    installs SadTalker mid-run (e.g. dynamic pip install on a
    long-lived process) the worker must call
    :func:`reset_sadtalker_import_cache` before retrying.
    """
    global _SADTALKER_IMPORT_ERROR
    try:
        # Upstream SadTalker layout (v0.0.x → Audio2MotionModel):
        #   sadtalker/
        #     src/
        #       audio2motion/
        #         models.py::Audio2MotionModel
        # Cloning the repo + `pip install -e .` is documented in
        # `docs/MODEL_LICENSES.md` under "Audio-bridge neural
        # replacement".
        from sadtalker.src.audio2motion.models import Audio2MotionModel

        return Audio2MotionModel
    except ImportError as exc:
        _SADTALKER_IMPORT_ERROR = exc
        LOG.debug("SadTalker Audio2Motion import failed: %s", exc)
        return None


def reset_sadtalker_import_cache() -> None:
    """Drop the cached import failure so a retry can succeed."""
    global _SADTALKER_IMPORT_ERROR
    _SADTALKER_IMPORT_ERROR = None


def sadtalker_available() -> bool:
    """Return True iff the SadTalker audio-to-motion submodule imports.

    Cheap probe — does NOT instantiate the model (which downloads
    200+ MB of weights and might require a CUDA device).
    """
    return _try_import_sadtalker() is not None


def audio_to_3dmm(
    audio_path: Path,
    *,
    start_seconds: float,
    end_seconds: float,
    fps: int,
    checkpoint_dir: Optional[Path] = None,
) -> np.ndarray:
    """Run SadTalker Audio2Motion on an audio window.

    Args:
        audio_path: 16-bit PCM mono WAV. The audio_bridge DSP layer
            resamples to 16 kHz internally if needed, so this can be
            any sample rate SadTalker accepts (16 kHz canonical, 22 kHz
            OK).
        start_seconds: window start (inclusive).
        end_seconds: window end (exclusive).
        fps: video fps of the chunk we'll render the deltas against.
        checkpoint_dir: directory where SadTalker checkpoints live. If
            None, SadTalker's default ``~/.cache/torch`` path is used.
            The Heyavatar checkpoint_manager is wired to populate this
            dir from the weight manifest in :mod:`providers.liveportrait.checkpoint_manager`.

    Returns:
        ``(T, 53)`` ``float32`` numpy array — the first ``50`` columns
        are 3DMM expression coefficients and the last ``3`` are jaw
        pose coefficients. ``T`` is ``round((end - start) * fps)``.

    Raises:
        RuntimeError: SadTalker is not importable, or the inference
            raised. The caller (:mod:`providers.liveportrait.audio_bridge.bridge`)
            surfaces this as ``EngineState.DEGRADED``.
        ValueError: ``coefs.shape[0] != expected_frames`` — SadTalker
            returned a tensor whose temporal length does not match
            our window. We deliberately do NOT silently resample; a
            drift on SadTalker's shape contract is the bug.
    """
    Audio2MotionModel = _try_import_sadtalker()
    if Audio2MotionModel is None:
        raise RuntimeError(
            "HEYAVATAR_AUDIO_BRIDGE_BACKEND=neural requires SadTalker's "
            f"audio2motion submodule; the import failed: "
            f"{_SADTALKER_IMPORT_ERROR}. Install with "
            "`pip install -e \".[audio-bridge-neural]\"` on the worker "
            "image (see docs/MODEL_LICENSES.md)."
        )

    # Build/load the model. SadTalker uses torch and CUDA; an import
    # succeeds on CPU boxes but ``forward`` will fail at first
    # ``.cuda()`` call. Surface that as a clear RuntimeError so the
    # engine transitions to DEGRADED.
    try:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Audio2MotionModel.load_default(checkpoint_dir=checkpoint_dir)
        model = model.to(device)
        model.eval()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"SadTalker Audio2MotionModel.load failed: "
            f"{type(exc).__name__}: {exc}. Verify the weight manifest "
            f"is populated (see providers.liveportrait.checkpoint_manager)."
        ) from exc

    # Read the audio window: re-use the DSP resampler so we don't
    # duplicate the WAV→numpy plumbing.
    from providers.liveportrait.audio_bridge.dsp import _read_wav_mono_16bit
    from providers.liveportrait.audio_bridge.dsp import _linear_resample

    samples, source_sr = _read_wav_mono_16bit(audio_path)
    target_sr = 16000
    resampled = _linear_resample(samples, source_sr, target_sr)
    start_idx = int(round(start_seconds * target_sr))
    end_idx = int(round(end_seconds * target_sr))
    start_idx = max(0, start_idx)
    end_idx = min(len(resampled), end_idx)
    audio_slice = np.asarray(
        resampled[start_idx:end_idx], dtype=np.float32
    ) / 32768.0  # int16 → [-1, 1]

    expected_frames = max(1, int(round((end_seconds - start_seconds) * fps)))
    if audio_slice.size == 0:
        # Silence / empty window. Return neutral coefs so the render
        # proceeds with a closed-mouth pose.
        exp = np.zeros((expected_frames, 50), dtype=np.float32)
        jaw = np.zeros((expected_frames, 3), dtype=np.float32)
        return np.concatenate([exp, jaw], axis=1)

    try:
        with torch.no_grad():
            wav_tensor = torch.from_numpy(audio_slice).unsqueeze(0).to(device)
            output = model(wav_tensor)  # shape (1, T_sadtalker, 53)
            coefs = output.squeeze(0).detach().cpu().numpy().astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"SadTalker Audio2Motion forward raised: "
            f"{type(exc).__name__}: {exc}."
        ) from exc

    # SadTalker's T can disagree with our expected T (it operates at
    # its own internal frame rate, typically 25 fps canonical).
    # Silent resampling would mask future drift on the upstream
    # shape contract — RAISE loudly instead so the bridge layer and
    # the operator both see SadTalker is the source of truth.
    # The bridge layer keeps a defensive ``ValueError`` check as a
    # backstop in case a future SadTalker version returns a 2-D
    # ``(53,)`` single-frame tensor.
    if coefs.shape[0] != expected_frames:
        raise ValueError(
            f"SadTalker Audio2Motion returned {coefs.shape[0]} frames; "
            f"expected {expected_frames}. Do NOT silently resample here — "
            f"a contract drift on the upstream model is the bug, not the "
            f"shape mismatch."
        )

    if coefs.shape[-1] != 53:
        # Some SadTalker checkpoints emit 50 (no jaw) — pad with zeros.
        if coefs.shape[-1] == 50:
            jaw_pad = np.zeros((coefs.shape[0], 3), dtype=np.float32)
            coefs = np.concatenate([coefs, jaw_pad], axis=1)
        else:
            raise RuntimeError(
                f"SadTalker audio2motion returned unexpected last-dim "
                f"{coefs.shape[-1]}; expected 50 or 53."
            )

    return coefs.astype(np.float32, copy=False)
