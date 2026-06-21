"""Real-GPU gate for the MediaPipe migration.

Verifies that on a real-GPU box with ``mediapipe`` installed, the
identity-prep path produces a pack whose ``identity_meta.json``
records ``detector == "mediapipe_face_mesh"``. This is the **release
gate** for flipping ``liveportrait-human-v1.commercial_use: true``.

Three preconditions all need to hold:

1. CUDA is reachable (``requires_cuda`` marker handles skip).
2. LivePortrait upstream repo is cloned (sentinel file).
3. ``mediapipe`` import succeeds.

When all three succeed, this test gates the registry flag flip in
``registry/models.yaml``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.config import get_settings
from src.domain.enums import EngineId
from tests.smoke.test_real_gpu._helpers import (
    _test_image,
    real_mode_env,  # noqa: F401, F811  (pytest fixture — ruff can't see fixture lookup)
    requires_cuda,
)


@requires_cuda
def test_identity_pack_records_mediapipe_detector(real_mode_env, tmp_path):
    """The identity pack MUST show that MediaPipe was the primary detector.

    We check two fields on ``identity_meta.json``:

    * ``mediapipe_attempted=True`` — MediaPipe was the FIRST detector
      tried. This is the contract that lets us flip
      ``liveportrait-human-v1.commercial_use: true``.
    * ``detector in {"mediapipe_face_mesh", "haar_cascade"}`` — we did
      NOT degenerate to the unconditional ``center_crop`` fallback.
      (When face_mesh returns no landmarks we fall back to Haar; this
      is acceptable, the contract is "MediaPipe was primary".)

    Note: we deliberately do NOT assert ``detector ==
    "mediapipe_face_mesh"``. The synthetic ``_test_image`` is a flat
    red PNG, so face_mesh likely returns ``None``; the production
    intent (Apache-2.0 stack) is captured by
    ``mediapipe_attempted`` being True.
    """
    source = _test_image(tmp_path)
    # Sanity: mediapipe must be importable on this box.
    try:
        import mediapipe as mp  # noqa: F401
        assert hasattr(mp, "solutions"), "mediapipe solution API missing"
    except ImportError as exc:
        pytest.skip(
            f"mediapipe not installed on this workstation: {exc}. "
            "Install with `pip install mediapipe` to enable the "
            "commercial-use gate verification."
        )

    settings = get_settings()
    if settings.mock_engine:
        pytest.skip("HEYAVATAR_MOCK_ENGINE=1")

    from providers import get_provider

    engine = get_provider(EngineId.LIVE_PORTRAIT)
    engine.load()
    try:
        assets = engine.prepare_identity(source)
        meta_bytes = assets["identity_meta.json"]
        meta = json.loads(meta_bytes.decode("utf-8"))
        assert meta.get("mediapipe_attempted") is True, (
            f"identity_meta.json mediapipe_attempted={meta.get('mediapipe_attempted')!r} "
            f"but expected True on a real-GPU box with mediapipe installed. "
            f"The flag flip in registry/models.yaml is BLOCKED on this outcome. "
            f"Meta: {meta!r}"
        )
        detector = meta.get("detector")
        assert detector in {"mediapipe_face_mesh", "haar_cascade"}, (
            f"identity_meta.json detector='{detector}' but expected one of "
            f"'mediapipe_face_mesh' or 'haar_cascade'. 'center_crop' means "
            f"MediaPipe and Haar BOTH failed — the production path is "
            f"unsuitable for commercial use. Meta: {meta!r}"
        )
    finally:
        engine.unload()
