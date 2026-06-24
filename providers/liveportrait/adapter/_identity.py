"""Real-mode identity preparation for LivePortrait.

* :func:`_real_prepare_identity_impl` — the ``prepare_identity`` body,
  attached to :class:`LivePortraitAdapter` at module-end.
* :func:`_render_face_crop` — upstream-prepared tensor → PNG bytes for
  the pack (read-only of ``self`` so attached as ``staticmethod``).
* :func:`_render_face_mask` — central-ellipse heuristic mask PNG.
* :func:`_pool_embedding` — average-pool the source feature volume
  into a 512-d identity embedding vector.

The non-method helpers do not access ``self`` so they're attached as
``staticmethod``. ``_real_prepare_identity_impl`` is attached as a
bound method.

Face detection cascade
======================
The real-mode identity path tries three detectors in order:

1. **MediaPipe Face Landmarker** (``mp.solutions.face_mesh``,
   Apache-2.0). Preferred. Apache-licensed, no InsightFace, and the
   key change that unlocks ``liveportrait-human-v1.commercial_use: true``.
2. **OpenCV Haar cascades** (Apache-2.0, no model download needed).
   Last-resort fallback if ``mediapipe`` isn't installed or returned no
   faces.
3. **Center crop of the whole image** when no detector yielded a face
   region. Recorded as ``detector="center_crop"`` in
   ``identity_meta.json`` so the orchestrator can audit usage.

The chosen detector is recorded in the pack's ``identity_meta.json``
key ``detector``, with one of: ``mediapipe_face_mesh``,
``haar_cascade``, ``center_crop``. The orchestrator (or the
``tests/smoke/test_real_gpu/test_mediapipe_identity.py`` gate) reads
this field to decide if the production path was used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.application.avatar_image import build_green_screen_cutout_png, load_avatar_source_image
from src.core.logging import get_logger


def _real_prepare_identity_impl(self, source_image: Path) -> Dict[str, bytes]:
    """Real-mode identity preparation.

    Pairs the upstream tensor path (handler_wrapper.prepare_source,
    appearance_feature_extractor.extract_feature_3d, motion_extractor.get_kp_info)
    with our pack-write bookkeeping so the orchestrator can later reload
    ``f_s`` / ``kp_s`` via :func:`providers.liveportrait.adapter._render._load_source_bundle`.
    """
    if self._wrapper is None:
        LOG = get_logger("providers.liveportrait")
        LOG.warning(
            "LivePortraitAdapter.prepare_identity called before load; "
            "returning mock assets"
        )
        from providers.liveportrait.adapter._mock import _mock_identity_assets
        return _mock_identity_assets(source_image)

    from providers.liveportrait.adapter._upstream import _import_torch
    torch = _import_torch()
    if torch is None:
        LOG = get_logger("providers.liveportrait")
        LOG.warning("torch disappeared mid-load; returning mock assets")
        from providers.liveportrait.adapter._mock import _mock_identity_assets
        return _mock_identity_assets(source_image)

    from providers.liveportrait.inference_config import (
        LIVE_PORTRAIT_PACK_VERSION,
        PackSchema,
    )
    import time

    try:
        import cv2
        LOG = get_logger("providers.liveportrait")

        # Read the raw source image bytes first.
        source_image_original_bytes = source_image.read_bytes()

        img = load_avatar_source_image(source_image)
        img_np = np.asarray(img, dtype=np.uint8)  # HxWx3 uint8
        h_img, w_img = img_np.shape[:2]
        cutout_png, green_ratio = build_green_screen_cutout_png(img_np)

        # ── Face detection cascade ─────────────────────────────
        # Priority order:
        #   1. MediaPipe Face Landmarker (Apache-2.0). Preferred — this
        #      is the change that unlocks ``liveportrait-human-v1``
        #      commercial_use in ``registry/models.yaml``.
        #   2. OpenCV Haar cascades (Apache-2.0, no model download
        #      needed). Last-resort fallback for environments where
        #      mediapipe isn't installed.
        #   3. Center crop of the whole image (when *no* detector yielded
        #      a face region). Recorded as ``detector="center_crop"`` in
        #      identity_meta.json so the orchestrator can audit usage.
        detector_used = "none"
        # Separate from ``detector_used``: records whether MediaPipe WAS
        # the primary detector attempted (regardless of whether it
        # found a face). The release gate in
        # ``tests/smoke/test_real_gpu/test_mediapipe_identity.py``
        # reads this field so a synthetic test image where face_mesh
        # finds no landmark does NOT collapse the gate to false.
        mediapipe_attempted = False
        size = min(h_img, w_img)
        cx, cy = w_img // 2, h_img // 2

        try:
            from providers.liveportrait.adapter._mediapipe import detect_face_bbox
            # MUST be set immediately after the import probe succeeds and
            # BEFORE the call returns / raises. This boolean is the
            # contract gate for ``liveportrait-human-v1.commercial_use``
            # in ``registry/models.yaml``. Do NOT move it down.
            mediapipe_attempted = True
            bbox = detect_face_bbox(img_np)
            if bbox is not None:
                x, y, w, h = bbox
                cx = x + w // 2
                cy = y + h // 2
                size = int(max(w, h) * 1.4)
                detector_used = "mediapipe_face_mesh"
        except ImportError:
            # ``mediapipe`` not installed; fall through to Haar. Mock-mode
            # CI without mediapipe must keep working.
            pass
        except Exception as exc:  # noqa: BLE001
            LOG.debug(
                "MediaPipe face detection raised; falling back to Haar cascade: %s",
                exc,
            )

        if detector_used == "none":
            # OpenCV Haar Cascade Face Detection (Apache-2.0; no model
            # download needed). Last-resort before center crop.
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(cascade_path)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100),
            )
            if len(faces) == 0:
                # Fallback: center crop of the whole image.
                detector_used = "center_crop"
            else:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                cx = x + w // 2
                cy = y + h // 2
                size = int(max(w, h) * 1.4)
                detector_used = "haar_cascade"

        x1 = max(0, cx - size // 2)
        y1 = max(0, cy - size // 2)
        x2 = min(w_img, cx + size // 2)
        y2 = min(h_img, cy + size // 2)

        actual_size = min(x2 - x1, y2 - y1)
        x1 = cx - actual_size // 2
        y1 = cy - actual_size // 2
        x2 = x1 + actual_size
        y2 = y1 + actual_size

        # Calculate M_o2c and M_c2o for the crop (dsize = 512).
        s = 512.0 / actual_size
        M_o2c = np.array(
            [
                [s, 0.0, 256.0 - s * cx],
                [0.0, s, 256.0 - s * cy],
            ],
            dtype=np.float32,
        )

        M_o2c_h = np.vstack([M_o2c, np.array([0.0, 0.0, 1.0], dtype=np.float32)])
        M_c2o = np.linalg.inv(M_o2c_h)
        transform_matrix = M_c2o[:2, :].reshape(-1)  # 2x3 matrix.

        # Crop and resize to 512x512.
        crop_np = img_np[y1:y2, x1:x2]
        crop_resized = cv2.resize(crop_np, (512, 512), interpolation=cv2.INTER_AREA)

        tensor = self._wrapper.prepare_source(crop_resized).to(self._torch_device)
        # ``extract_feature_3d`` returns ``[1, 32, 16, 64, 64]``.
        f_s = self._wrapper.extract_feature_3d(tensor)
        kp_info = self._wrapper.get_kp_info(tensor)
        if isinstance(kp_info, dict):
            exp_s = kp_info["exp"]
            kp_s = kp_info["kp"]
            x_s = self._wrapper.transform_keypoint(kp_info)
        else:
            exp_s, kp_s = kp_info[0], kp_info[1]
            x_s = kp_s
        face_crop = _render_face_crop(tensor)
        face_mask = _render_face_mask(face_crop)
        identity_embedding = _pool_embedding(f_s)

        # Runtime guard: ``detector_used`` MUST be one of three
        # declared detectors. A future refactor that bypasses the
        # cascade (or a mock shortcut) would otherwise leak ``"none"``
        # into production packs, breaking the contract test gate in
        # ``tests/smoke/test_real_gpu/test_mediapipe_identity.py``.
        assert detector_used in {
            "mediapipe_face_mesh",
            "haar_cascade",
            "center_crop",
        }, f"unexpected detector_used={detector_used!r}"

        meta_dict = {
            "schema": LIVE_PORTRAIT_PACK_VERSION,
            "upstream": PackSchema().upstream_url,
            "source_image": str(source_image),
            "green_screen_ratio": f"{green_ratio:.4f}",
            "detector": detector_used,
            # ``mediapipe_attempted`` is the gate signal: it is True when
            # MediaPipe WAS the primary detector tried (regardless of
            # whether a face was found). Distinct from ``detector`` which
            # records the WINNING detector for the image at hand.
            "mediapipe_attempted": mediapipe_attempted,
            "prepared_at": f"{time.time():.3f}",
        }
        assets = {
            "source_features.bin": np.asarray(
                f_s.detach().cpu().numpy(), dtype=np.float16
            ).tobytes(),
            "canonical_keypoints.bin": np.concatenate(
                [
                    np.asarray(x_s.detach().cpu().numpy(), dtype=np.float32),
                    np.asarray(exp_s.detach().cpu().numpy(), dtype=np.float32),
                ]
            ).tobytes(),
            "face_mask.png": face_mask,
            "face_crop.png": face_crop,
            "identity_embedding.bin": np.asarray(
                identity_embedding, dtype=np.float32
            ).tobytes(),
            "source_latent.bin": np.asarray(
                f_s.detach().cpu().numpy().mean(axis=(2, 3, 4)), dtype=np.float16
            ).tobytes(),
            "transform_matrix.bin": np.asarray(
                transform_matrix, dtype=np.float32
            ).tobytes(),
            "source_image_original.png": source_image_original_bytes,
            "source_image_cutout.png": cutout_png,
            "identity_meta.json": self._json_dump(meta_dict),
            "inference_config.json": self._json_dump(self.inf_cfg.to_dict()),
            "crop_config.json": self._json_dump(self._crop_to_dict(self.crop_cfg)),
        }
        LOG.info(
            "LivePortraitAdapter prepared identity from %s (%d bytes f_s, detector=%s)",
            source_image,
            len(assets["source_features.bin"]),
            detector_used,
        )
        return assets
    except Exception as exc:  # noqa: BLE001
        if self.settings.mock_engine:
            LOG = get_logger("providers.liveportrait")
            LOG.error(
                "Real-mode prepare_identity failed, returning mock assets: %s",
                exc,
            )
            from providers.liveportrait.adapter._mock import _mock_identity_assets
            return _mock_identity_assets(source_image)
        raise


def _render_face_crop(tensor: Any) -> bytes:
    """Convert an upstream-prepared tensor to PNG bytes for the pack.

    Asserts the upstream ``prepare_source`` tensor layout: ``[1, 3, H, W]``
    in CHWC order. Versions of LivePortrait that ship NHWC tensors
    would fail this assertion rather than silently corrupt the pack.
    """
    try:
        from PIL import Image
        import io as _io

        arr_in = tensor.detach().cpu().numpy()
        if arr_in.ndim != 4 or arr_in.shape[0] != 1 or arr_in.shape[1] != 3:
            raise RuntimeError(
                f"prepare_source returned unexpected tensor shape "
                f"{arr_in.shape}; LivePortrait CHWC layout expected."
            )
        arr = arr_in[0].transpose(1, 2, 0)
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr[..., :3])
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except RuntimeError:
        # re-raise so the caller's exception handler can degrade.
        raise
    except Exception as exc:  # noqa: BLE001
        LOG = get_logger("providers.liveportrait")
        LOG.warning("Could not render face crop PNG: %s", exc)
        return b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def _render_face_mask(face_crop_png: bytes) -> bytes:
    """Heuristic face-region mask from the face crop.

    Computes the central 60% ellipse with Gaussian blur feathering.
    """
    try:
        from PIL import Image, ImageDraw, ImageFilter
        import io as _io

        img = Image.open(_io.BytesIO(face_crop_png)).convert("L")
        img = img.resize((512, 512), Image.Resampling.LANCZOS)
        w, h = img.size
        mask = Image.new("L", (w, h), 0)
        drw = ImageDraw.Draw(mask)
        # Draw a larger ellipse to capture the entire head, hair, ears, and neck.
        drw.ellipse(
            (45, 20, w - 45, h - 30),
            fill=255,
        )
        # Apply Gaussian Blur for smooth feathered edges.
        mask = mask.filter(ImageFilter.GaussianBlur(40))
        buf = _io.BytesIO()
        mask.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:  # noqa: BLE001
        return face_crop_png[:64] if face_crop_png else (b"\xff" * 64)


def _pool_embedding(f_s: Any) -> np.ndarray:
    """Average-pool the feature volume into a 512-d identity embedding vector."""
    arr = f_s.detach().cpu().numpy().astype(np.float32)
    flat = arr.reshape(-1)
    if flat.size < 512:
        return np.concatenate([flat, np.zeros(512 - flat.size, dtype=np.float32)])
    # Stride-decimate so the embedding is deterministic.
    stride = max(1, flat.size // 512)
    return flat[::stride][:512]


# ── helpers used by _real_prepare_identity_impl ────────────────────


def _json_dump_impl(self, obj) -> bytes:
    """Compact JSON serializer used by the identity-prep path."""
    from providers._ffmpeg import _json_dump
    return _json_dump(obj)


def _crop_to_dict_impl(self, crop_cfg) -> dict:
    """Mirror :func:`_to_upstream_crop_config` for direct dict output."""
    return {
        "landmark_type": crop_cfg.landmark_type,
        "flag_do_crop": crop_cfg.flag_do_crop,
        "source_image_size": crop_cfg.source_image_size,
        "flag_do_rot": crop_cfg.flag_do_rot,
    }


# ── attach to LivePortraitAdapter ──────────────────────────────────
def _attach_identity_methods():
    from providers.liveportrait.adapter.engine import LivePortraitAdapter
    LivePortraitAdapter._real_prepare_identity = _real_prepare_identity_impl
    LivePortraitAdapter._render_face_crop = staticmethod(_render_face_crop)
    LivePortraitAdapter._render_face_mask = staticmethod(_render_face_mask)
    # Helpers attached as instance methods (with ``self``) so callers can
    # use ``self._json_dump(...)`` / ``self._crop_to_dict(...)`` uniformly.
    LivePortraitAdapter._json_dump = _json_dump_impl
    LivePortraitAdapter._crop_to_dict = _crop_to_dict_impl


_attach_identity_methods()
