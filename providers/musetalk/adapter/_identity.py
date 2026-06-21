"""Real-mode identity preparation for MuseTalk.

Detects the source face, aligns it to a 256×256 canonical crop, runs
that crop through the VAE to a 4-channel latent, and emits the pack
assets the orchestrator expects.

The methods here are attached to :class:`MuseTalkAdapter` at
import time (see end of this file) so callers can use
``adapter._real_prepare_identity(...)`` etc. via the instance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from providers.musetalk.adapter._upstream import _import_torch
from src.core.logging import get_logger


def _real_prepare_identity_impl(self, source_image: Path) -> dict:
    """Real-mode: face detect → align → VAE encode → pack assets."""
    torch = _import_torch()
    if torch is None:
        raise RuntimeError("PyTorch not available for prepare_identity")

    # ── 1. Load image ─────────────────────────────────────────
    from PIL import Image
    img = Image.open(source_image).convert("RGB")
    img_np = np.asarray(img, dtype=np.uint8)

    # ── 2. Face detection + alignment (MediaPipe or upstream) ──
    face_crop, alignment_matrix = self._detect_and_align(img_np)
    face_crop_png = self._encode_png(face_crop)

    # ── 3. VAE encode ─────────────────────────────────────────
    face_tensor = (
        torch.from_numpy(face_crop.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(self._torch_device)
    )
    source_latent = self._vae_encode(face_tensor, torch)

    # ── 4. Identity embedding ─────────────────────────────────
    latent_flat = source_latent.detach().cpu().numpy().flatten()
    stride = max(1, latent_flat.size // 512)
    embedding = latent_flat[::stride][:512].astype(np.float32)

    # ── 5. Face mask (central ellipse heuristic) ──────────────
    face_mask_png = self._render_face_mask(face_crop_png)

    get_logger(__name__).info(
        "MuseTalkAdapter prepared identity from %s (crop %dx%d)",
        source_image,
        face_crop.shape[1],
        face_crop.shape[0],
    )
    return {
        "source_latent.bin": source_latent.detach()
        .cpu()
        .numpy()
        .astype(np.float16)
        .tobytes(),
        "face_crop.png": face_crop_png,
        "face_mask.png": face_mask_png,
        "identity_embedding.bin": embedding.tobytes(),
        "alignment_matrix.bin": alignment_matrix.astype(np.float32).tobytes(),
    }


def _detect_and_align_impl(
    self, img_np: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect face and produce a 256×256 aligned crop.

    Tries upstream MuseTalk's face detector first, then MediaPipe.
    Falls back to a centre crop if neither is available.
    """
    # Try upstream detector.
    if self._pipeline is not None:
        try:
            detector = getattr(self._pipeline, "face_detector", None)
            if detector is not None:
                if callable(detector):
                    return detector(img_np, target_size=256)
                detect_fn = getattr(detector, "detect_and_align", None)
                if detect_fn is not None:
                    return detect_fn(img_np, target_size=256)
        except Exception as exc:
            get_logger(__name__).debug("Upstream face detector failed: %s", exc)

    # Try MediaPipe.
    try:
        import mediapipe as mp
        mp_face = mp.solutions.face_detection
        with mp_face.FaceDetection(
            model_selection=1, min_detection_confidence=0.7
        ) as fd:
            results = fd.process(img_np)
            if results.detections:
                det = results.detections[0]
                h, w = img_np.shape[:2]
                bbox = det.location_data.relative_bounding_box
                cx = int((bbox.xmin + bbox.width / 2) * w)
                cy = int((bbox.ymin + bbox.height / 2) * h)
                size = int(max(bbox.width * w, bbox.height * h) * 1.4)
                return self._crop_and_resize(
                    img_np, cx, cy, size, 256
                )
    except ImportError:
        get_logger(__name__).debug("MediaPipe not installed for face detection")
    except Exception as exc:
        get_logger(__name__).debug("MediaPipe face detection failed: %s", exc)

    # Fallback: centre crop.
    get_logger(__name__).warning("No face detector available; using centre crop")
    h, w = img_np.shape[:2]
    size = min(h, w)
    cx, cy = w // 2, h // 2
    return self._crop_and_resize(img_np, cx, cy, size, 256)


def _crop_and_resize(
    img: np.ndarray,
    cx: int,
    cy: int,
    size: int,
    target_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop a square region around (cx, cy) and resize to target."""
    from PIL import Image

    half = size // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(img.shape[1], cx + half)
    y2 = min(img.shape[0], cy + half)
    crop = img[y1:y2, x1:x2]
    pil_crop = Image.fromarray(crop).resize(
        (target_size, target_size), Image.LANCZOS
    )
    aligned = np.asarray(pil_crop)
    # Identity alignment matrix as fallback.
    matrix = np.array(
        [1.0, 0.0, float(-x1), 0.0, 1.0, float(-y1)], dtype=np.float32
    )
    return aligned, matrix


def _vae_encode_impl(self, face_tensor, torch) -> "torch.Tensor":
    """Encode face tensor through VAE to get source latent.

    In mock mode, returns a deterministic random latent for
    testability. In real mode, raises if VAE is unavailable.
    """
    if self._pipeline is not None:
        try:
            vae = getattr(self._pipeline, "vae", None)
            if vae is not None and hasattr(vae, "encode"):
                mu, logvar = vae.encode(face_tensor)
                return mu
        except Exception as exc:
            get_logger(__name__).warning("VAE encode via upstream failed: %s", exc)
            if not self.settings.mock_engine:
                raise RuntimeError(
                    "MuseTalk VAE encode failed in real mode"
                ) from exc

    if self.settings.mock_engine:
        get_logger(__name__).warning("VAE not available; using random latent")
        return torch.randn(
            1, 4, 64, 64, device=self._torch_device, dtype=torch.float32
        ) * 0.01

    raise RuntimeError(
        "MuseTalk VAE is unavailable in real mode. "
        "Install upstream MuseTalk or set HEYAVATAR_MOCK_ENGINE=1."
    )


def _render_face_mask(self, face_crop_png: bytes) -> bytes:
    """Central-ellipse heuristic mask, used as a fallback / mock."""
    try:
        from PIL import Image, ImageDraw
        import io as _io

        img = Image.open(_io.BytesIO(face_crop_png)).convert("L")
        w, h = img.size
        mask = Image.new("L", (w, h), 0)
        drw = ImageDraw.Draw(mask)
        cx, cy = w // 2, h // 2
        drw.ellipse(
            (
                cx - w // 3,
                cy - h // 3,
                cx + w // 3,
                cy + h // 3,
            ),
            fill=255,
        )
        buf = _io.BytesIO()
        mask.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return face_crop_png[:64] if face_crop_png else b"\xff" * 64


def _encode_png(arr: np.ndarray) -> bytes:
    """Encode a HxWx3 (uint8) numpy array as PNG bytes."""
    from PIL import Image
    import io as _io

    img = Image.fromarray(arr)
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── bind to MuseTalkAdapter ───────────────────────────────────────
# Imported lazily so we attach to the class defined in :mod:`engine`
# without ever importing the engine module twice.
def _attach_identity_methods():
    from providers.musetalk.adapter.engine import MuseTalkAdapter
    MuseTalkAdapter._real_prepare_identity = _real_prepare_identity_impl
    MuseTalkAdapter._detect_and_align = _detect_and_align_impl
    MuseTalkAdapter._crop_and_resize = staticmethod(_crop_and_resize)
    MuseTalkAdapter._vae_encode = _vae_encode_impl
    MuseTalkAdapter._encode_png = staticmethod(_encode_png)
    MuseTalkAdapter._render_face_mask = _render_face_mask


_attach_identity_methods()
