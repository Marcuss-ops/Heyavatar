"""Unit tests for the MediaPipe Face Landmarker helper.

Verifies the contract of
:func:`providers.liveportrait.adapter._mediapipe.detect_face_bbox`:

* Returns ``None`` when MediaPipe finds no landmarks.
* When multiple faces are detected, returns the bounding box with
  the largest area (in normalised image coordinates).
* A processing exception propagates so the caller can fall back to
  OpenCV Haar cascades.

End-to-end (real-GPU) coverage lives in
``tests/smoke/test_real_gpu/test_mediapipe_identity.py`` — these tests
exercise the helper in isolation by injecting a stub ``mediapipe``
module into ``sys.modules`` so torch / CUDA / real mediapipe are NOT
needed.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest


def _install_stub_mediapipe(monkeypatch, face_mesh_cls):
    """Inject a stub ``mediapipe`` module that exposes ``FaceMesh = face_mesh_cls``."""
    stub = types.SimpleNamespace(FaceMesh=face_mesh_cls)
    mp_mod = types.ModuleType("mediapipe")
    mp_mod.solutions = types.SimpleNamespace(face_mesh=stub)
    monkeypatch.setitem(sys.modules, "mediapipe", mp_mod)


def _make_landmark(x: float, y: float):
    return types.SimpleNamespace(x=x, y=y)


def _make_face_landmarks(landmark_pairs):
    """Wrap a list of ``(x, y)`` pairs in a ``.landmark`` attribute."""
    return types.SimpleNamespace(
        landmark=[_make_landmark(x, y) for x, y in landmark_pairs]
    )


# ---------------------------------------------------------------------------
# Contract: returns None when no faces are found.
# ---------------------------------------------------------------------------


def test_detect_face_bbox_returns_none_when_no_faces(monkeypatch):
    class _NoFaces:
        def __init__(self, *_a, **_k) -> None:
            pass
        def __enter__(self):
            return self
        def __exit__(self, *_a):
            return False
        def process(self, _img):
            return types.SimpleNamespace(multi_face_landmarks=None)

    _install_stub_mediapipe(monkeypatch, _NoFaces)
    from providers.liveportrait.adapter._mediapipe import detect_face_bbox

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    assert detect_face_bbox(img) is None


# ---------------------------------------------------------------------------
# Contract: when several faces are detected, returns the largest.
# ---------------------------------------------------------------------------


def test_detect_face_bbox_largest_face_wins(monkeypatch):
    small_face = _make_face_landmarks(
        [(0.0, 0.0), (0.05, 0.0), (0.05, 0.05), (0.0, 0.05)]
    )
    big_face = _make_face_landmarks(
        [(0.1, 0.1), (0.6, 0.1), (0.6, 0.6), (0.1, 0.6)]
    )

    class _MultiFace:
        def __init__(self, *_a, **_k) -> None:
            pass
        def __enter__(self):
            return self
        def __exit__(self, *_a):
            return False
        def process(self, _img):
            return types.SimpleNamespace(
                multi_face_landmarks=[small_face, big_face]
            )

    _install_stub_mediapipe(monkeypatch, _MultiFace)
    from providers.liveportrait.adapter._mediapipe import detect_face_bbox

    # Image 100x100 → BIG span [0.1, 0.6] × [0.1, 0.6] → bbox (10, 10, 50, 50).
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = detect_face_bbox(img)
    assert bbox is not None
    x, y, w, h = bbox
    assert (x, y, w, h) == (10, 10, 50, 50)


# ---------------------------------------------------------------------------
# Contract: runtime exceptions propagate so the caller can fall back.
# ---------------------------------------------------------------------------


def test_detect_face_bbox_propagates_runtime_error(monkeypatch):
    class _Boom:
        def __init__(self, *_a, **_k):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *_a):
            return False
        def process(self, _img):
            raise RuntimeError("simulated mediapipe crash")

    _install_stub_mediapipe(monkeypatch, _Boom)
    from providers.liveportrait.adapter._mediapipe import detect_face_bbox

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="simulated mediapipe crash"):
        detect_face_bbox(img)


