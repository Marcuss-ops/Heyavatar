"""render_clean_composite — CLI tool for the clean compositing pipeline.

Usage::

    python -m tools.avatar_assets.render_clean_composite \\
      --body        tmp_debug_test/explain_both/body.mp4 \\
      --face        tmp_debug_test/musetalk_test/lipsynced_face.mp4 \\
      --face-mask   tmp_debug_test/explain_both/face_mask.mp4 \\
      --neck-mask   tmp_debug_test/explain_both/neck_mask.mp4 \\
      --transforms  tmp_debug_test/explain_both/face_transforms.npz \\
      --audio       tmp_debug_test/speech.wav \\
      --output      captures/final_clean.mp4

Exit codes:
  0 — COMPLETED (QC passed)
  1 — compositing, encoding, or QC failure
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from contracts.compositor import CompositeRequest
from contracts.quality_checker import QCRequest
from src.pipeline import OpenCVFaceCompositor
from src.quality.exceptions import CompositeError, EncodingError, QualityError
from src.quality.video_quality import VideoQualityChecker


# ─────────────────────────────────────────────────────────────────────────────
# Stage helpers
# ─────────────────────────────────────────────────────────────────────────────

def _composite(
    body: Path,
    face: Path,
    face_mask: Path,
    neck_mask: Path,
    transforms: Path,
    composited_path: Path,
    debug: bool,
) -> None:
    """Run OpenCV compositor.  Raises CompositeError on failure."""
    compositor = OpenCVFaceCompositor()
    request = CompositeRequest(
        body_video=body,
        generated_face_video=face,
        face_mask_video=face_mask,
        neck_mask_video=neck_mask,
        face_transforms=transforms,
        output_path=composited_path,
        debug=debug,
    )
    result = compositor.composite(request)
    print(
        f"  [COMPOSITING] {result.frames_processed} frames, "
        f"{result.dropped_frames} dropped, "
        f"avg_mask_area={result.average_mask_area:.3f}"
    )


def _mux_audio(composited: Path, audio: Path, final: Path) -> None:
    """Mux video + audio with ffmpeg.  Raises EncodingError on failure."""
    cmd = [
        "ffmpeg",
        "-i", str(composited),
        "-i", str(audio),
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        "-y",
        str(final),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise EncodingError(
            f"ffmpeg mux failed (exit {proc.returncode}): "
            + proc.stderr.decode("utf-8", errors="replace")[-400:]
        )


def _quality_check(final: Path, audio: Path) -> None:
    """Run VideoQualityChecker.  Raises QualityError if QC fails."""
    checker = VideoQualityChecker()
    result = checker.check_quality(QCRequest(video_path=final, audio_path=audio))

    # Always print the metric table
    print()
    print("  QC RESULTS")
    print(f"    status             : {result.status}")
    print(f"    debug_green_ratio  : {result.debug_green_ratio:.6f}")
    print(f"    black_frame_ratio  : {result.black_frame_ratio:.4f}")
    print(f"    duration_delta_ms  : {result.duration_delta_ms:.1f}")
    print(f"    frames_expected    : {result.frames_expected}")
    print(f"    frames_actual      : {result.frames_actual}")
    print(f"    invalid_transforms : {result.invalid_transforms}")
    for w in result.warnings:
        print(f"    [WARN] {w}")
    for e in result.errors:
        print(f"    [ERROR] {e}")

    if not result.passed:
        raise QualityError(result.status, result)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Clean face compositing pipeline with mandatory QC gate."
    )
    parser.add_argument("--body",       required=True, help="Body video path (.mp4)")
    parser.add_argument("--face",       required=True, help="Lipsynced face video (.mp4)")
    parser.add_argument("--face-mask",  required=True, help="Face mask video (.mp4)")
    parser.add_argument("--neck-mask",  required=True, help="Neck mask video (.mp4)")
    parser.add_argument("--transforms", required=True, help="Face transforms (.npz)")
    parser.add_argument("--audio",      required=True, help="Audio file (.wav)")
    parser.add_argument("--output",     required=True, help="Output video path")
    parser.add_argument("--job-id",     default="",    help="Optional job identifier for output layout")
    parser.add_argument("--debug",      action="store_true", help="Write debug preview videos")
    args = parser.parse_args(argv)

    output_path = Path(args.output)
    job_id = args.job_id or output_path.stem

    # Resolve output layout:
    #   captures/<job_id>/runtime/composited.mp4
    #   captures/<job_id>/runtime/final.mp4   ← args.output is redirected here
    runtime_dir = output_path.parent / job_id / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    composited_path = runtime_dir / "composited.mp4"
    final_path      = runtime_dir / "final.mp4"

    # Write a result summary alongside the output
    result_path = runtime_dir.parent / "result.json"

    print(f"[render_clean_composite] job_id={job_id}")
    print(f"  body       : {args.body}")
    print(f"  face       : {args.face}")
    print(f"  face-mask  : {args.face_mask}")
    print(f"  neck-mask  : {args.neck_mask}")
    print(f"  transforms : {args.transforms}")
    print(f"  audio      : {args.audio}")
    print(f"  output     : {final_path}")
    print(f"  debug      : {args.debug}")
    print()

    try:
        # ── 1. COMPOSITING ────────────────────────────────────────────────────
        print("[1/3] COMPOSITING...")
        _composite(
            body       = Path(args.body),
            face       = Path(args.face),
            face_mask  = Path(args.face_mask),
            neck_mask  = Path(args.neck_mask),
            transforms = Path(args.transforms),
            composited_path = composited_path,
            debug      = args.debug,
        )

        # ── 2. ENCODING (audio mux) ───────────────────────────────────────────
        print("[2/3] ENCODING (mux audio)...")
        _mux_audio(composited_path, Path(args.audio), final_path)
        print(f"  Muxed video saved to: {final_path}")

        # ── 3. QUALITY CHECK ──────────────────────────────────────────────────
        print("[3/3] QUALITY CHECK...")
        _quality_check(final_path, Path(args.audio))

    except CompositeError as exc:
        status = "FAILED_COMPOSITING"
        print(f"\nFAILED: {exc}")
        _write_result(result_path, status, str(exc), final_path)
        return 1

    except EncodingError as exc:
        status = "FAILED_ENCODING"
        print(f"\nFAILED: {exc}")
        _write_result(result_path, status, str(exc), final_path)
        return 1

    except QualityError as exc:
        _write_result(result_path, exc.status, str(exc), final_path)
        return 1

    # ── Success ───────────────────────────────────────────────────────────────
    print(f"\nQC PASSED -> {final_path}")
    _write_result(result_path, "COMPLETED", "", final_path)
    return 0


def _write_result(path: Path, status: str, error: str, output: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {"status": status, "output": str(output)}
    if error:
        data["error"] = error
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    sys.exit(main())
