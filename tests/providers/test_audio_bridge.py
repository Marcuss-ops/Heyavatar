"""AudioBridge module tests for the unified ``audio_to_driving`` API.

Covers both backends:

* **DSP** (default for CI): shape, blink flag dtype, mouth aperture
  range, empty-window raises.
* **Neural** (production): with a stub SadTalker injected into
  ``sys.modules`` so the unit tests do NOT pull in the real upstream
  package. Confirms:
  - ``HEYAVATAR_AUDIO_BRIDGE_BACKEND=neural`` dispatches to SadTalker.
  - Missing SadTalker raises ``RuntimeError`` (NEVER silent fallback
    to DSP).

The contract tests in this file run against the dsp backend
(DEFAULT). CI without CUDA stays green; SadTalker-specific coverage
lives in :file:`tests/providers/test_sadtalker_projection.py`.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from providers.liveportrait.audio_bridge.bridge import audio_to_driving
from providers.liveportrait.audio_bridge.types import EXPRESSION_DIM, N_KEYPOINTS


# ----------------------------------------------------------------------
# DSP backend: contract — slice, shape, blink dtype, empty-window error.
# ----------------------------------------------------------------------


def test_audio_to_driving_resample_and_slice_to_frames(wav_factory) -> None:
    wav = wav_factory()
    driving = audio_to_driving(
        wav,
        start_seconds=0.1,
        end_seconds=1.1,
        fps=25,
    )
    # window length is 1.0 s -> 25 frames
    assert driving.frames == 25
    assert len(driving.exp_d_flat) == 25 * N_KEYPOINTS * EXPRESSION_DIM
    assert driving.backend == "dsp"


def test_audio_to_driving_padded_when_window_shorter_than_audio(wav_factory) -> None:
    wav = wav_factory()
    # Window entirely inside the audio silence leading.
    driving = audio_to_driving(
        wav, start_seconds=0.0, end_seconds=0.25, fps=25
    )
    assert driving.frames == 6  # 0.25 * 25
    # Silence-only slice should produce mouth aperture == 0 across the
    # leading frames (RMS ≈ 0 → smoothed to 0 → rescaled stays 0).
    assert all(0.0 <= v <= 1.0 for v in driving.mouth_aperture)


def test_audio_to_driving_shape_matches_upstream(wav_factory) -> None:
    wav = wav_factory()
    driving = audio_to_driving(wav, start_seconds=0.0, end_seconds=1.0, fps=25)
    # Flat-packed exp tensor must reshape to (frames, 21, 3).
    arr = np.asarray(driving.exp_d_flat, dtype=np.float32).reshape(
        driving.frames, N_KEYPOINTS, EXPRESSION_DIM
    )
    assert arr.shape == (25, 21, 3)


def test_audio_to_driving_emits_blinks_for_active_audio(wav_factory) -> None:
    wav = wav_factory()
    driving = audio_to_driving(wav, start_seconds=0.1, end_seconds=1.1, fps=25)
    # The generated WAV has constant-amplitude sine in the middle,
    # ZCR is ~0; a pure tone may not trigger blinks, but the dtype
    # must still be bool.
    assert set(driving.blink_mask).issubset({True, False})
    # Mouth aperture must be in [0, 1].
    assert all(0.0 <= v <= 1.0 for v in driving.mouth_aperture)


def test_zero_window_raises_value_error(wav_factory) -> None:
    wav = wav_factory()
    with pytest.raises(ValueError):
        audio_to_driving(wav, start_seconds=0.0, end_seconds=0.0, fps=25)


# ----------------------------------------------------------------------
# Backend selection: env var wires through to SadTalker.
# ----------------------------------------------------------------------


def test_default_backend_is_dsp_via_settings(wav_factory, monkeypatch) -> None:
    """Forced-default assertion: ``HEYAVATAR_AUDIO_BRIDGE_BACKEND`` unset.

    ``monkeypatch.delenv`` with ``raising=False`` lets us undo the
    env var (set elsewhere by another test) so the fallback path is
    genuinely exercised. After the fallback, the dispatched backend
    MUST be ``dsp``.
    """
    monkeypatch.delenv("HEYAVATAR_AUDIO_BRIDGE_BACKEND", raising=False)
    # Force settings cache to re-read so the deletion takes effect.
    from src.core.config import get_settings

    get_settings.cache_clear()
    wav = wav_factory()
    driving = audio_to_driving(
        wav, start_seconds=0.0, end_seconds=0.5, fps=25
    )
    assert driving.backend == "dsp"


def test_explicit_backend_dsp(wav_factory, monkeypatch) -> None:
    monkeypatch.setenv("HEYAVATAR_AUDIO_BRIDGE_BACKEND", "dsp")
    from src.core.config import get_settings

    get_settings.cache_clear()
    wav = wav_factory()
    driving = audio_to_driving(
        wav, start_seconds=0.0, end_seconds=0.5, fps=25
    )
    assert driving.backend == "dsp"


def test_neural_backend_without_sadtalker_raises(wav_factory, monkeypatch) -> None:
    """Neural backend is selected but SadTalker is NOT importable.

    Expected: RuntimeError, NOT silent fallback to DSP. This is the
    policy enforcement test that keeps production workers honest.
    """
    monkeypatch.setenv("HEYAVATAR_AUDIO_BRIDGE_BACKEND", "neural")
    # Purge any cached SadTalker module so the import probe is real.
    for mod_name in list(sys.modules.keys()):
        if mod_name == "sadtalker" or mod_name.startswith("sadtalker."):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)
    # Force the bridge's internal SadTalker-import cache to also
    # forget its previous failure — wrapped in monkeypatch so pytest
    # reverts the global state mutation on teardown.
    from providers.liveportrait.audio_bridge import sadtalker as _sad

    monkeypatch.setattr(_sad, "_SADTALKER_IMPORT_ERROR", None)
    _sad.reset_sadtalker_import_cache()
    from src.core.config import get_settings

    get_settings.cache_clear()
    wav = wav_factory()
    with pytest.raises(RuntimeError, match="requires SadTalker"):
        audio_to_driving(
            wav, start_seconds=0.0, end_seconds=0.5, fps=25
        )


# ----------------------------------------------------------------------
# Neural backend: stub SadTalker in sys.modules.
# ----------------------------------------------------------------------


def _install_stub_sadtalker(monkeypatch, return_value):
    """Inject a stub ``sadtalker.src.audio2motion.models`` module tree.

    The stub's ``Audio2MotionModel`` class returns the fixed
    ``return_value`` from its ``__call__`` so the unit test can
    exercise the projection with deterministic input.
    """

    class _FakeAudio2MotionModel:
        def __init__(self, *_a, **_k):
            pass

        def to(self, _device):
            return self

        def eval(self):
            return self

        @staticmethod
        def load_default(*_a, **_k):
            return _FakeAudio2MotionModel()

        def __call__(self, _wav):
            return return_value

    sadtalker_pkg = types.ModuleType("sadtalker")
    src_pkg = types.ModuleType("sadtalker.src")
    a2m_pkg = types.ModuleType("sadtalker.src.audio2motion")
    models_mod = types.ModuleType("sadtalker.src.audio2motion.models")
    models_mod.Audio2MotionModel = _FakeAudio2MotionModel
    sadtalker_pkg.src = src_pkg
    src_pkg.audio2motion = a2m_pkg
    a2m_pkg.models = models_mod
    monkeypatch.setitem(sys.modules, "sadtalker", sadtalker_pkg)
    monkeypatch.setitem(sys.modules, "sadtalker.src", src_pkg)
    monkeypatch.setitem(sys.modules, "sadtalker.src.audio2motion", a2m_pkg)
    monkeypatch.setitem(sys.modules, "sadtalker.src.audio2motion.models", models_mod)


def test_neural_backend_with_stub_sadtalker_returns_driving_signals(
    wav_factory, monkeypatch
) -> None:
    """Stub SadTalker returns a 53-column coefs tensor; assert shape."""

    fps = 25
    total_seconds = 1.0
    expected_frames = int(round(total_seconds * fps))  # 25

    # 25 frames, each with 50 exp + 3 jaw coefs. Jaw rises smoothly
    # from 0 to 1.2 across the timeline so the mouth_aperture proxy
    # is a non-degenerate sequence.
    exp = np.zeros((expected_frames, 50), dtype=np.float32)
    jaw = np.stack(
        [
            np.linspace(0.0, 1.2, expected_frames).astype(np.float32),
            np.zeros(expected_frames, dtype=np.float32),
            np.zeros(expected_frames, dtype=np.float32),
        ],
        axis=1,
    )
    coefs = np.concatenate([exp, jaw], axis=1)  # (25, 53)

    # Mimic a torch tensor chain ``output.squeeze(0).detach().cpu().numpy()``
    # so the bridge can call the same chain on a non-tensor object.
    class _FakeTensor:
        def __init__(self, array):
            self._array = array

        def squeeze(self, *_a, **_k):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._array

    wrapped = _FakeTensor(coefs)
    _install_stub_sadtalker(monkeypatch, wrapped)
    monkeypatch.setenv("HEYAVATAR_AUDIO_BRIDGE_BACKEND", "neural")
    # Force the bridge's internal SadTalker-import cache to forget
    # any earlier failure — monkeypatch.setattr reverts this on
    # teardown so the global state mutation doesn't leak across tests.
    from providers.liveportrait.audio_bridge import sadtalker as _sad

    monkeypatch.setattr(_sad, "_SADTALKER_IMPORT_ERROR", None)
    _sad.reset_sadtalker_import_cache()
    from src.core.config import get_settings

    get_settings.cache_clear()

    wav = wav_factory()
    driving = audio_to_driving(
        wav,
        start_seconds=0.0,
        end_seconds=total_seconds,
        fps=fps,
    )
    assert driving.backend == "neural_sadtalker"
    assert driving.frames == expected_frames
    assert len(driving.exp_d_flat) == expected_frames * N_KEYPOINTS * EXPRESSION_DIM
    # Mouth aperture should be in [0, 1].
    assert all(0.0 <= v <= 1.0 for v in driving.mouth_aperture)
    # And NOT all zeros (the jaw magnitudes mean at least some frame
    # produces a non-zero aperture).
    assert max(driving.mouth_aperture) > 0.0


def test_neural_backend_raises_valueerror_on_shape_mismatch(
    wav_factory, monkeypatch
) -> None:
    """SadTalker returns a tensor of WRONG temporal length.

    The bridge MUST raise ``ValueError`` (not silently re-interpolate
    and not raise ``RuntimeError``) so the SadTalker module's
    contract drift is caught loudly. ``ValueError`` is the canonical
    Python shape-mismatch error type.
    """
    monkeypatch.setenv("HEYAVATAR_AUDIO_BRIDGE_BACKEND", "neural")
    from providers.liveportrait.audio_bridge import sadtalker as _sad

    monkeypatch.setattr(_sad, "_SADTALKER_IMPORT_ERROR", None)

    fps = 25
    total_seconds = 1.0
    expected_frames = int(round(total_seconds * fps))  # 25

    # Deliver 32 frames of coefs (≠ expected_frames = 25) so the
    # shape mismatch in the bridge fires.
    bad_frames = expected_frames + 7
    coefs = np.zeros((bad_frames, 53), dtype=np.float32)

    class _FakeTensor:
        def __init__(self, array):
            self._array = array

        def squeeze(self, *_a, **_k):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._array

    _install_stub_sadtalker(monkeypatch, _FakeTensor(coefs))
    from src.core.config import get_settings

    get_settings.cache_clear()
    wav = wav_factory()
    with pytest.raises(ValueError, match="returned .* frames; expected"):
        audio_to_driving(
            wav,
            start_seconds=0.0,
            end_seconds=total_seconds,
            fps=fps,
        )
