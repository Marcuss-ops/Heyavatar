"""Engine load + identity prep in real (non-mock) mode.

Verifies that with ``HEYAVATAR_MOCK_ENGINE=0`` and a real GPU, the
LivePortrait adapter can:

* be instantiated via :func:`providers.get_provider`
* successfully ``load()`` onto the GPU
* report health as ``idle`` or ``loading`` with ``mock_mode=False``
* :meth:`prepare_identity` produces real arithmetic feature volumes —
  not the byte-level stub the mock engine returns.
"""

from __future__ import annotations

import pytest

from src.core.config import get_settings
from src.domain.enums import EngineId
from tests.smoke.test_real_gpu._helpers import (
    _test_image,
    real_mode_env,
    requires_cuda,
)


@requires_cuda
def test_engine_loads_in_real_mode(real_mode_env, workdir, tmp_path):
    """Verify the LivePortrait engine loads successfully with GPU."""
    settings = get_settings()
    if settings.mock_engine:
        pytest.skip("HEYAVATAR_MOCK_ENGINE=1")

    from providers import get_provider

    engine = get_provider(EngineId.LIVE_PORTRAIT)
    assert engine.engine_id == EngineId.LIVE_PORTRAIT

    try:
        engine.load()
        health = engine.health()
        print(f"\nEngine state after load: {health.state.value}")
        assert health.state.value in ("idle", "loading"), (
            f"Engine should be IDLE or LOADING after load(), got {health.state.value}"
        )
        assert health.mock_mode is False, "Should be in real mode"
        assert health.vram_used_mb >= 0
    finally:
        engine.unload()


@requires_cuda
def test_prepare_identity_real_mode(real_mode_env, workdir, tmp_path):
    """Verify prepare_identity() produces real (non-mock) assets."""
    settings = get_settings()
    if settings.mock_engine:
        pytest.skip("HEYAVATAR_MOCK_ENGINE=1")

    source = _test_image(tmp_path)

    from providers import get_provider

    engine = get_provider(EngineId.LIVE_PORTRAIT)
    engine.load()
    try:
        assets = engine.prepare_identity(source)
        assert isinstance(assets, dict)
        # Real mode must produce the full feature volume (not mock bytes).
        assert "source_features.bin" in assets
        f_s_bytes = assets["source_features.bin"]
        assert len(f_s_bytes) > 1000, (
            f"source_features.bin too small ({len(f_s_bytes)} B) — "
            "likely mock mode or degraded"
        )
        # Should produce a real face crop (not the mock random noise).
        assert "face_crop.png" in assets
        # The canonical keypoints should be present.
        assert "canonical_keypoints.bin" in assets
        print(
            f"\nIdentity prepared: {len(assets)} assets, "
            f"source_features={len(f_s_bytes)} B"
        )
    finally:
        engine.unload()
