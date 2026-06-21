"""MediaPipe Face Landmarker adapter for LivePortrait identity preparation.

Thin wrapper around :mod:`mediapipe.solutions.face_mesh` (Apache-2.0)
that returns the ``(x, y, w, h)`` bounding box of the largest detected
face in the input image, or ``None`` if no face is found.

Used by LivePortrait's identity-prep path as the **primary** detector
when ``mediapipe`` is installed; the adapter falls back to OpenCV Haar
cascades if MediaPipe import fails or detection returns zero faces.

Why ``face_mesh`` (not ``face_detection`` or ``FaceLandmarker`` Tasks)?
========================================================================
* ``face_mesh`` IS the canonical "MediaPipe Face Landmarker" — 468
  landmarks per face, with the bounding box derivable from those
  landmarks' ``x`` / ``y`` extremes.
* It uses the legacy ``mp.solutions`` API, so no extra ``.task`` model
  download is required. ``pip install mediapipe`` is enough.
* The landmark set is future-proof: if we ever want 5-landmark affine
  alignment (eyes, nose, mouth corners) instead of bbox-crop, the
  landmarks are already there.

License
=======
Both ``mediapipe`` and OpenCV are Apache-2.0. This module is the gate
that unlocks ``liveportrait-human-v1.commercial_use=true`` in
``registry/models.yaml``; the flag is currently ``false`` and only
flips after the contract tests run successfully on a real GPU.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def detect_face_bbox(img_np: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return ``(x, y, w, h)`` of the largest detected face, or ``None``.

    Args:
        img_np: ``HxWx3`` ``uint8`` numpy array in **RGB** ordering
            (matches what PIL emits when calling ``np.asarray`` on an
            "RGB" image and what `LivePortraitAdapter._real_prepare_identity`
            passes in).

    Returns:
        ``(x, y, w, h)`` tuple of ints in image pixel coordinates for
        the bounding box of the largest face detected. ``None`` if no
        faces are found.

    Raises:
        ImportError: ``mediapipe`` is not installed.
        RuntimeError: media pipe processing raised unexpectedly and
            the caller would lose the avatar pack entirely — callers
            SHOULD catch this and fall back to OpenCV Haar cascades.
    """
    import mediapipe as mp  # optional dependency (probe-detected at runtime)

    h, w = img_np.shape[:2]
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=10,
        min_detection_confidence=0.5,
    ) as fm:
        results = fm.process(img_np)
    if not results.multi_face_landmarks:
        return None

    best_bbox: Optional[Tuple[int, int, int, int]] = None
    max_area = 0
    for face_landmarks in results.multi_face_landmarks:
        xs = [lm.x for lm in face_landmarks.landmark]
        ys = [lm.y for lm in face_landmarks.landmark]
        x_min = int(min(xs) * w)
        x_max = int(max(xs) * w)
        y_min = int(min(ys) * h)
        y_max = int(max(ys) * h)
        fwidth = x_max - x_min
        fheight = y_max - y_min
        area = fwidth * fheight
        if area > max_area:
            max_area = area
            best_bbox = (x_min, y_min, fwidth, fheight)
    return best_bbox
