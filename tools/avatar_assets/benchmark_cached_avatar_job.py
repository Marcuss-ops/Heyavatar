"""End-to-end benchmark for a cached-avatar job runtime directory."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contracts.quality_checker import QCRequest
from src.motion.benchmark import benchmark_pose_track_path
from src.quality.video_quality import VideoQualityChecker


@dataclass(slots=True)
class CachedAvatarBenchmarkReport:
    status: str
    final_path: str
    motion_track_path: str
    gesture_variety: float
    presence_score: float
    motion_energy: float
    transition_density: float
    steady_pose_ratio: float
    qc_status: str
    debug_green_ratio: float
    black_frame_ratio: float
    duration_delta_ms: float
    gpu_seconds: float
    wall_seconds: float
    gpu_seconds_per_output_minute: float


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def benchmark_cached_avatar_job(runtime_dir: Path, *, audio_path: Path | None = None) -> CachedAvatarBenchmarkReport:
    metrics = _load_json(runtime_dir / "metrics.json")
    result = _load_json(runtime_dir / "result.json")
    final_path = runtime_dir / "final.mp4"
    if not final_path.is_file():
        final_path = Path(result.get("final_path") or metrics.get("final_path") or final_path)
    motion_track_path_raw = metrics.get("motion_track_path", "")
    motion_track_path = Path(motion_track_path_raw) if motion_track_path_raw else runtime_dir / "motion_track.npz"

    if motion_track_path.is_file():
        motion_benchmark = benchmark_pose_track_path(motion_track_path)
    else:
        motion_benchmark = benchmark_pose_track_path(final_path.with_suffix(".npz")) if final_path.with_suffix(".npz").is_file() else None

    chosen_audio = audio_path
    if chosen_audio is None:
        chosen_audio = runtime_dir / "audio.wav"
        if not chosen_audio.is_file():
            chosen_audio = runtime_dir.parent / "audio.wav"

    if chosen_audio is not None and chosen_audio.is_file() and final_path.is_file():
        qc = VideoQualityChecker()
        qc_result = qc.check_quality(QCRequest(video_path=final_path, audio_path=chosen_audio))
    else:
        qc_result = type(
            "_QC",
            (),
            {
                "status": "SKIPPED",
                "debug_green_ratio": 0.0,
                "black_frame_ratio": 0.0,
                "duration_delta_ms": 0.0,
            },
        )()

    return CachedAvatarBenchmarkReport(
        status=str(result.get("status") or metrics.get("status") or "UNKNOWN"),
        final_path=str(final_path),
        motion_track_path=str(motion_track_path),
        gesture_variety=getattr(motion_benchmark, "gesture_variety", 0.0),
        presence_score=getattr(motion_benchmark, "presence_score", 0.0),
        motion_energy=getattr(motion_benchmark, "motion_energy", 0.0),
        transition_density=getattr(motion_benchmark, "transition_density", 0.0),
        steady_pose_ratio=getattr(motion_benchmark, "steady_pose_ratio", 1.0),
        qc_status=qc_result.status,
        debug_green_ratio=qc_result.debug_green_ratio,
        black_frame_ratio=qc_result.black_frame_ratio,
        duration_delta_ms=qc_result.duration_delta_ms,
        gpu_seconds=float(metrics.get("gpu_seconds", 0.0)),
        wall_seconds=float(metrics.get("wall_seconds", 0.0)),
        gpu_seconds_per_output_minute=float(metrics.get("gpu_seconds_per_output_minute", 0.0)),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark a cached-avatar runtime directory.")
    parser.add_argument("--runtime-dir", required=True, type=Path, help="Job runtime directory containing metrics.json")
    parser.add_argument("--audio", type=Path, default=None, help="Optional audio path for QC")
    parser.add_argument("--output", type=Path, default=Path("cached_avatar_benchmark.json"), help="JSON output path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = benchmark_cached_avatar_job(args.runtime_dir, audio_path=args.audio)
    payload = {
        "status": report.status,
        "final_path": report.final_path,
        "motion_track_path": report.motion_track_path,
        "gesture_variety": report.gesture_variety,
        "presence_score": report.presence_score,
        "motion_energy": report.motion_energy,
        "transition_density": report.transition_density,
        "steady_pose_ratio": report.steady_pose_ratio,
        "qc_status": report.qc_status,
        "debug_green_ratio": report.debug_green_ratio,
        "black_frame_ratio": report.black_frame_ratio,
        "duration_delta_ms": report.duration_delta_ms,
        "gpu_seconds": report.gpu_seconds,
        "wall_seconds": report.wall_seconds,
        "gpu_seconds_per_output_minute": report.gpu_seconds_per_output_minute,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
