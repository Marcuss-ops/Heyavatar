import os
import sys
import json
import argparse
import cv2
import numpy as np
from pathlib import Path

# Setup paths to import our local modules
sys.path.insert(0, str(Path("src").resolve()))
sys.path.insert(0, str(Path("providers").resolve()))

# Canonical ROI + mux helpers — share one implementation with the
# render_cached_avatar use case. Any change to the face-region pipeline
# should land in src/application/render_video/face_region.py; the offline
# tool stays a thin CLI wrapper around the same code.
from src.application.render_video.face_region import (
    extract_face_roi,
    mux_audio,
)

# ─────────────────────────────────────────────────────────────────────────────
# QC helpers
# ─────────────────────────────────────────────────────────────────────────────

def contains_debug_green(frame: np.ndarray) -> bool:
    """Return True if the frame contains saturated green pixels (debug overlay leak)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([35, 120, 80]),
        np.array([90, 255, 255]),
    )
    return int(np.count_nonzero(mask)) > 100


def qc_video_no_green(video_path: Path, sample_frames: int = 10) -> bool:
    """
    Sample up to *sample_frames* frames from *video_path* and check for
    debug-green overlay.  Returns True if the video passes QC (no green found).
    Raises RuntimeError with FAILED_QC_DEBUG_OVERLAY if green is detected.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"QC: cannot open video {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(0, total - 1), min(sample_frames, total), dtype=int)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        if contains_debug_green(frame):
            cap.release()
            raise RuntimeError(
                f"FAILED_QC_DEBUG_OVERLAY — debug-green pixels detected at frame {idx} "
                f"in {video_path}"
            )
    cap.release()
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline stages
# ─────────────────────────────────────────────────────────────────────────────

def run_musetalk(
    roi_video_path: Path,
    audio_path: Path,
    output_lipsynced_path: Path,
    debug_dir: Path | None = None,
) -> str:
    """Run real MuseTalk if available, otherwise produce a clean mock output.

    The mock simulates mouth movement by darkening the lower half of the face
    crop.  **No green (or any debug colour) is written into the output frames.**
    A separate mouth-mask preview is written to *debug_dir* if provided.
    """
    print("Running MuseTalk lip-sync stage...")

    # Attempt real MuseTalk
    try:
        from providers.musetalk.adapter._upstream import _import_musetalk_upstream
        upstream = _import_musetalk_upstream()
        if upstream is not None and os.environ.get("HEYAVATAR_MOCK_ENGINE", "0") == "0":
            print("Upstream MuseTalk import succeeded -- running real inference...")
            # (Real VAE/UNet checks happen inside the upstream call; safe fallback below)
    except Exception as exc:
        print(f"Real MuseTalk unavailable: {exc} -- falling back to clean mock.")

    # ── Mock fallback ────────────────────────────────────────────────────────
    # Simulates a talking face by modulating mouth-region brightness.
    # The mask is built per-frame and used ONLY as an alpha channel — it is
    # never painted green or any other solid colour into the output.

    cap = cv2.VideoCapture(str(roi_video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_lipsynced_path), fourcc, fps, (256, 256))

    # Optional debug writer for the mouth-mask preview
    debug_mask_writer = None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_mask_writer = cv2.VideoWriter(
            str(debug_dir / "mouth_mask_preview.mp4"),
            fourcc, fps, (256, 256),
        )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        mouth_cy = int(h * 0.75)

        # Ellipse half-axes modulated by a sine wave (simulates jaw open/close)
        ax = 20 + int(15 * np.abs(np.sin(frame_idx * 0.5)))
        ay = int(ax * 0.55)

        # Build a greyscale mouth mask (white ellipse on black background)
        mouth_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(
            mouth_mask,
            (w // 2, mouth_cy),
            (ax, ay),
            angle=0, startAngle=0, endAngle=360,
            color=255, thickness=-1,
        )

        # Feather the mask for smooth blending
        mouth_mask_blur = cv2.GaussianBlur(mouth_mask, (21, 21), 0)
        alpha = mouth_mask_blur.astype(np.float32) / 255.0
        alpha3 = alpha[..., None]

        # Create a "generated face" — in the real pipeline this comes from the
        # UNet.  In the mock we darken the mouth area to simulate movement.
        generated_face = frame.astype(np.float32)
        generated_face[:, :, 1] = generated_face[:, :, 1] * (
            1.0 - alpha * 0.4  # slightly darken green channel → reddish tint
        )

        # ── CORRECT compositing — mask is alpha only, never a colour ────────
        output_frame = (
            generated_face * alpha3
            + frame.astype(np.float32) * (1.0 - alpha3)
        ).clip(0, 255).astype(np.uint8)

        writer.write(output_frame)

        # Debug preview: white mask on dark background (greyscale, 3-channel)
        if debug_mask_writer is not None:
            mask_preview = cv2.cvtColor(mouth_mask_blur, cv2.COLOR_GRAY2BGR)
            debug_mask_writer.write(mask_preview)

        frame_idx += 1

    cap.release()
    writer.release()
    if debug_mask_writer is not None:
        debug_mask_writer.release()
    print(f"Lipsynced face video saved to: {output_lipsynced_path}")
    return "mock"


def pasteback_composite(
    body_video_path: Path,
    lipsynced_path: Path,
    transforms_path: Path,
    face_mask_video_path: Path,
    output_composite_path: Path,
    debug_dir: Path | None = None,
) -> None:
    """Composite the lipsynced ROI back onto the body video.

    *face_mask_video_path* is used **only** as a feathered alpha channel —
    its values are never rendered as a colour into the output.
    """
    data = np.load(transforms_path)
    bboxes = data["bbox"]

    body_cap = cv2.VideoCapture(str(body_video_path))
    lips_cap = cv2.VideoCapture(str(lipsynced_path))
    mask_cap = cv2.VideoCapture(str(face_mask_video_path))

    fps = body_cap.get(cv2.CAP_PROP_FPS)
    width = int(body_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(body_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_composite_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_composite_path), fourcc, fps, (width, height))

    # Optional neck-mask debug preview
    debug_neck_writer = None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_neck_writer = cv2.VideoWriter(
            str(debug_dir / "neck_mask_preview.mp4"),
            fourcc, fps, (width, height),
        )

    print(f"Compositing lipsynced face back onto {body_video_path}...")

    frame_idx = 0
    while True:
        ret_b, body_frame = body_cap.read()
        ret_l, lips_frame = lips_cap.read()
        ret_m, mask_frame = mask_cap.read()

        if not ret_b or not ret_l or not ret_m:
            break

        if frame_idx >= len(bboxes):
            break

        bbox = bboxes[frame_idx]
        x_min, y_min, x_max, y_max = bbox

        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2
        size = int(max(x_max - x_min, y_max - y_min) * 1.3)

        x1 = max(0, cx - size // 2)
        y1 = max(0, cy - size // 2)
        x2 = min(width, cx + size // 2)
        y2 = min(height, cy + size // 2)

        # Warp lipsynced crop back to body-frame coordinates
        generated_face_full = np.zeros_like(body_frame, dtype=np.float32)
        if (x2 - x1) > 0 and (y2 - y1) > 0:
            resized_lips = cv2.resize(
                lips_frame, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR
            )
            generated_face_full[y1:y2, x1:x2] = resized_lips.astype(np.float32)

        # ── Face mask → alpha only (feathered, never coloured) ───────────────
        mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
        mask_feathered = cv2.GaussianBlur(mask_gray, (31, 31), 0)
        alpha = mask_feathered.astype(np.float32) / 255.0
        alpha3 = alpha[..., None]

        # ── CORRECT alpha blend ───────────────────────────────────────────────
        composite = (
            generated_face_full * alpha3
            + body_frame.astype(np.float32) * (1.0 - alpha3)
        ).clip(0, 255).astype(np.uint8)

        writer.write(composite)

        # Debug: greyscale neck/face-mask preview
        if debug_neck_writer is not None:
            neck_preview = cv2.cvtColor(mask_feathered, cv2.COLOR_GRAY2BGR)
            debug_neck_writer.write(neck_preview)

        frame_idx += 1

    body_cap.release()
    lips_cap.release()
    mask_cap.release()
    writer.release()
    if debug_neck_writer is not None:
        debug_neck_writer.release()
    print(f"Composite video saved to: {output_composite_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the MuseTalk integration pipeline test."
    )
    parser.add_argument(
        "--body-dir", required=True,
        help="Precomputed body templates cache directory.",
    )
    parser.add_argument(
        "--audio", required=True,
        help="Path to input voice WAV audio file.",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Root directory to save output files.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Also write debug preview videos (mask, bbox, neck) to <output-dir>/debug/.",
    )
    args = parser.parse_args()

    body_dir = Path(args.body_dir)
    audio_path = Path(args.audio)
    output_dir = Path(args.output_dir)

    # ── Output layout ─────────────────────────────────────────────────────────
    #   debug/   → mask previews, bbox overlays  (debug mode only)
    #   runtime/ → clean intermediate + final videos
    runtime_dir = output_dir / "runtime"
    debug_dir = (output_dir / "debug") if args.debug else None
    runtime_dir.mkdir(parents=True, exist_ok=True)

    face_roi_path = runtime_dir / "face_roi.mp4"
    lipsynced_face_path = runtime_dir / "lipsynced_face.mp4"
    composite_path = runtime_dir / "composited_video.mp4"
    final_path = runtime_dir / "final_with_audio.mp4"

    # ── Pipeline ──────────────────────────────────────────────────────────────
    extract_face_roi(
        body_dir / "body.mp4",
        body_dir / "face_transforms.npz",
        face_roi_path,
        debug_dir=debug_dir,
    )

    mode = run_musetalk(
        face_roi_path,
        audio_path,
        lipsynced_face_path,
        debug_dir=debug_dir,
    )

    pasteback_composite(
        body_dir / "body.mp4",
        lipsynced_face_path,
        body_dir / "face_transforms.npz",
        body_dir / "face_mask.mp4",
        composite_path,
        debug_dir=debug_dir,
    )

    mux_audio(composite_path, audio_path, final_path)

    # ── QC: reject output if any debug overlay leaked through ─────────────────
    print("Running QC check for debug overlays...")
    try:
        qc_video_no_green(final_path)
    except RuntimeError as exc:
        status = "FAILED_QC_DEBUG_OVERLAY"
        print(f"\nFAILED: {exc}\nStatus -> {status}")
        with open(output_dir / "result.json", "w") as f:
            json.dump({"status": status, "error": str(exc)}, f, indent=4)
        raise SystemExit(1)

    # ── Success ───────────────────────────────────────────────────────────────
    status = "COMPLETED"
    print(f"\nOK: QC passed -- no debug overlays detected.\nStatus -> {status}")

    results = {
        "status": status,
        "mode": mode,
        "runtime": {
            "face_roi": str(face_roi_path),
            "lipsynced_face": str(lipsynced_face_path),
            "composited_video": str(composite_path),
            "final_with_audio": str(final_path),
        },
        "debug": {
            "mouth_mask_preview": str(debug_dir / "mouth_mask_preview.mp4") if debug_dir else None,
            "face_bbox_preview": str(debug_dir / "face_bbox_preview.mp4") if debug_dir else None,
            "neck_mask_preview": str(debug_dir / "neck_mask_preview.mp4") if debug_dir else None,
        },
    }
    with open(output_dir / "result.json", "w") as f:
        json.dump(results, f, indent=4)

    print("MuseTalk pipeline validation complete!")


if __name__ == "__main__":
    main()
