"""OpenCVFaceCompositor — frame-by-frame face compositing.

**Canonical location** (per Change 2 of
``docs/REPOSITORY_SLIMMING_PLAN.md`` §4): ``src/pipeline/compositor.py``.
The previous home at ``providers/compositing/opencv_face/compositor.py``
has been deleted; both the runtime path and the offline preview tool
import from here. Contract (``contracts.compositor.Compositor``) and
class name are unchanged.

Capabilities:

* feathered alpha blend (face_mask + neck_mask)
* per-channel color matching (mean-std normalisation)
* temporal smoothing (EMA on mask and bbox)
* strict debug/runtime separation

Pipeline per frame
------------------
1.  Read body frame
2.  Read generated face frame
3.  Read face_mask frame
4.  Read neck_mask frame
5.  Read bbox from face_transforms.npz
6.  Resize generated face to bbox crop area
7.  Color match generated face → body frame statistics
8.  Build feathered combined alpha (face + neck * 0.35)
9.  Apply temporal EMA smoothing to alpha
10. Alpha-blend: generated * alpha + body * (1 − alpha)
11. (debug only) Write overlay previews to debug/
12. Write clean frame to runtime output
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from contracts.compositor import CompositeRequest, CompositeResult, Compositor
from src.core.logging import get_logger
from src.quality.exceptions import CompositeError

LOG = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Color matching helper
# ─────────────────────────────────────────────────────────────────────────────

def match_mean_std(
    source: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Shift *source* pixel statistics towards *target* in the masked region.

    Operates per BGR channel.  Returns a float32 image (same shape as source).
    If the masked area is too small the source is returned unchanged.

    Args:
        source: float32 BGR image to correct (the generated face crop).
        target: float32 BGR image to match (the body frame in the blend area).
        mask:   float32 single-channel mask (0‥1) selecting the blend region.

    Returns:
        Colour-corrected version of *source* as float32.
    """
    result = source.copy()
    min_pixels = 50

    for ch in range(3):
        s_ch = source[:, :, ch]
        t_ch = target[:, :, ch]

        s_vals = s_ch[mask > 0.3]
        t_vals = t_ch[mask > 0.3]

        if len(s_vals) < min_pixels or len(t_vals) < min_pixels:
            continue

        s_mean, s_std = float(np.mean(s_vals)), float(np.std(s_vals))
        t_mean, t_std = float(np.mean(t_vals)), float(np.std(t_vals))

        if s_std < 1e-3:
            continue

        scale = t_std / s_std if s_std > 1e-3 else 1.0
        # Clamp scale to avoid blowout
        scale = float(np.clip(scale, 0.5, 2.0))
        result[:, :, ch] = (s_ch - s_mean) * scale + t_mean

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Debug overlay helper — called only when request.debug is True
# ─────────────────────────────────────────────────────────────────────────────

def _draw_debug_overlay(
    frame: np.ndarray,
    alpha: np.ndarray,
    bbox: Tuple[int, int, int, int],
    frame_idx: int,
) -> np.ndarray:
    """Return a copy of *frame* with alpha heat-map + bbox drawn on it.

    This function may use ANY colour because its output goes only to debug/.
    """
    dbg = frame.copy()
    # Heat-map of the combined alpha: blue=low, red=high
    alpha_u8 = (alpha * 255).clip(0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(alpha_u8, cv2.COLORMAP_JET)
    cv2.addWeighted(dbg, 0.7, heatmap, 0.3, 0, dbg)
    # Bounding box
    x1, y1, x2, y2 = bbox
    cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 200, 255), 2)
    cv2.putText(
        dbg, f"f{frame_idx}", (x1, max(0, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1,
    )
    return dbg


# ─────────────────────────────────────────────────────────────────────────────
# Main compositor
# ─────────────────────────────────────────────────────────────────────────────

class OpenCVFaceCompositor(Compositor):
    """Frame-by-frame compositor using OpenCV alpha blending.

    Parameters
    ----------
    ema_alpha:
        Weight of the current frame mask in the temporal EMA.
        Higher = less smoothing (faster response).  Range 0‥1.
    face_blur_ksize:
        Gaussian kernel size for feathering the face mask.
    neck_blur_ksize:
        Gaussian kernel size for feathering the neck mask.
    neck_weight:
        Maximum alpha contribution of the neck mask (0‥1).
    """

    def __init__(
        self,
        ema_alpha: float = 0.65,
        face_blur_ksize: int = 31,
        neck_blur_ksize: int = 51,
        neck_weight: float = 0.35,
    ) -> None:
        self._ema_alpha = float(np.clip(ema_alpha, 0.01, 1.0))
        self._face_blur_ksize = face_blur_ksize | 1   # must be odd
        self._neck_blur_ksize = neck_blur_ksize | 1
        self._neck_weight = float(np.clip(neck_weight, 0.0, 1.0))

    # ── public API ────────────────────────────────────────────────────────────

    def composite(self, request: CompositeRequest) -> CompositeResult:  # noqa: PLR0912, PLR0915
        """Run the full compositing pipeline.

        Raises
        ------
        CompositeError
            If a critical input cannot be opened or the pipeline fails.
        """
        self._validate_inputs(request)

        # Load transforms
        try:
            data = np.load(request.face_transforms)
        except Exception as exc:
            raise CompositeError(f"Cannot load face_transforms: {exc}") from exc

        bboxes: np.ndarray = data.get("bbox", data.get("bboxes", None))
        if bboxes is None:
            raise CompositeError(
                "face_transforms.npz must contain a 'bbox' array"
            )

        # Open video captures
        body_cap   = self._open_cap(request.body_video, "body_video")
        face_cap   = self._open_cap(request.generated_face_video, "generated_face_video")
        fmask_cap  = self._open_cap(request.face_mask_video, "face_mask_video")
        nmask_cap  = self._open_cap(request.neck_mask_video, "neck_mask_video")

        fps    = body_cap.get(cv2.CAP_PROP_FPS) or 25.0
        width  = int(body_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(body_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Prepare output directory layout
        runtime_dir = request.output_path.parent
        runtime_dir.mkdir(parents=True, exist_ok=True)

        debug_dir: Optional[Path] = None
        debug_writer: Optional[cv2.VideoWriter] = None
        if request.debug:
            debug_dir = runtime_dir.parent / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)

        fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
        writer  = cv2.VideoWriter(str(request.output_path), fourcc, fps, (width, height))
        if request.debug and debug_dir is not None:
            debug_writer = cv2.VideoWriter(
                str(debug_dir / "composite_debug.mp4"),
                fourcc, fps, (width, height),
            )

        # State for temporal EMA
        prev_alpha: Optional[np.ndarray] = None

        frames_processed = 0
        dropped_frames   = 0
        mask_area_sum    = 0.0

        try:
            for frame_idx in range(len(bboxes)):
                ret_b, body_frame  = body_cap.read()
                ret_f, face_frame  = face_cap.read()
                ret_m, fmask_frame = fmask_cap.read()
                ret_n, nmask_frame = nmask_cap.read()

                if not (ret_b and ret_f and ret_m and ret_n):
                    LOG.debug("Stream exhausted at frame %d", frame_idx)
                    break

                # ── 1. Decode bbox ────────────────────────────────────────────
                bbox = bboxes[frame_idx]
                if not np.all(np.isfinite(bbox)):
                    LOG.warning("Non-finite bbox at frame %d; dropping", frame_idx)
                    writer.write(body_frame)   # pass through body unchanged
                    dropped_frames += 1
                    frames_processed += 1
                    continue

                x_min, y_min, x_max, y_max = [int(v) for v in bbox]
                cx = (x_min + x_max) // 2
                cy = (y_min + y_max) // 2
                size = int(max(x_max - x_min, y_max - y_min) * 1.3)
                x1 = max(0, cx - size // 2)
                y1 = max(0, cy - size // 2)
                x2 = min(width,  cx + size // 2)
                y2 = min(height, cy + size // 2)

                crop_w = x2 - x1
                crop_h = y2 - y1
                if crop_w <= 0 or crop_h <= 0:
                    writer.write(body_frame)
                    dropped_frames += 1
                    frames_processed += 1
                    continue

                # ── 2. Resize generated face to bbox area ─────────────────────
                resized_face = cv2.resize(
                    face_frame, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR
                )

                # ── 3. Build full-frame generated canvas ──────────────────────
                generated_full = np.zeros_like(body_frame, dtype=np.float32)
                generated_full[y1:y2, x1:x2] = resized_face.astype(np.float32)

                # ── 4. Color match in the blend area ──────────────────────────
                face_mask_gray = cv2.cvtColor(fmask_frame, cv2.COLOR_BGR2GRAY)
                face_alpha_raw = face_mask_gray.astype(np.float32) / 255.0
                generated_full = match_mean_std(
                    generated_full,
                    body_frame.astype(np.float32),
                    face_alpha_raw,
                )

                # ── 5. Feathered alpha: face mask ──────────────────────────────
                face_blur = cv2.GaussianBlur(
                    face_mask_gray,
                    (self._face_blur_ksize, self._face_blur_ksize), 0,
                )
                face_alpha = face_blur.astype(np.float32) / 255.0

                # ── 6. Feathered alpha: neck mask (lighter weight) ─────────────
                neck_gray = cv2.cvtColor(nmask_frame, cv2.COLOR_BGR2GRAY)
                neck_blur = cv2.GaussianBlur(
                    neck_gray,
                    (self._neck_blur_ksize, self._neck_blur_ksize), 0,
                )
                neck_alpha = (neck_blur.astype(np.float32) / 255.0) * self._neck_weight

                # ── 7. Combined alpha: face dominates, neck fills transition ──
                combined_alpha = np.maximum(face_alpha, neck_alpha)
                combined_alpha = np.clip(combined_alpha, 0.0, 1.0)

                # ── 8. Temporal EMA smoothing ─────────────────────────────────
                if prev_alpha is None:
                    smoothed_alpha = combined_alpha
                else:
                    smoothed_alpha = (
                        self._ema_alpha * combined_alpha
                        + (1.0 - self._ema_alpha) * prev_alpha
                    )
                prev_alpha = smoothed_alpha.copy()

                alpha3 = smoothed_alpha[..., np.newaxis]

                # ── 9. Alpha blend — mask is ONLY alpha, never a colour ────────
                composite = (
                    generated_full * alpha3
                    + body_frame.astype(np.float32) * (1.0 - alpha3)
                ).clip(0, 255).astype(np.uint8)

                # ── 10. Write runtime output ───────────────────────────────────
                writer.write(composite)

                # ── 11. Debug output (optional) ────────────────────────────────
                if debug_writer is not None:
                    dbg_frame = _draw_debug_overlay(
                        composite, smoothed_alpha, (x1, y1, x2, y2), frame_idx
                    )
                    debug_writer.write(dbg_frame)

                mask_area_sum += float(np.mean(smoothed_alpha))
                frames_processed += 1

        except Exception as exc:
            raise CompositeError(f"Compositing failed at frame {frames_processed}: {exc}") from exc
        finally:
            body_cap.release()
            face_cap.release()
            fmask_cap.release()
            nmask_cap.release()
            writer.release()
            if debug_writer is not None:
                debug_writer.release()

        avg_area = mask_area_sum / frames_processed if frames_processed > 0 else 0.0
        LOG.info(
            "Compositing complete: {} frames, {} dropped, avg_mask_area={:.3f}",
            frames_processed, dropped_frames, avg_area,
        )

        return CompositeResult(
            output_path=request.output_path,
            frames_processed=frames_processed,
            dropped_frames=dropped_frames,
            average_mask_area=avg_area,
            debug_overlay_detected=False,   # runtime output never contains debug
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_inputs(request: CompositeRequest) -> None:
        missing = [
            str(p)
            for p in (
                request.body_video,
                request.generated_face_video,
                request.face_mask_video,
                request.neck_mask_video,
                request.face_transforms,
            )
            if not p.is_file()
        ]
        if missing:
            raise CompositeError(f"Missing input files: {missing}")

    @staticmethod
    def _open_cap(path: Path, name: str) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise CompositeError(f"Cannot open {name}: {path}")
        return cap


__all__ = ["OpenCVFaceCompositor", "match_mean_std"]
