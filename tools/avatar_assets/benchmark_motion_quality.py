"""Benchmark gesture presence and motion quality on extracted assets."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from contracts.quality_checker import QCRequest
from src.motion.benchmark import MotionBenchmarkResult, benchmark_pose_track
from src.motion.pose_graph import PoseGraphTrack
from src.quality.video_quality import VideoQualityChecker


@dataclass(slots=True)
class MotionBenchmarkReport:
    gesture_variety: float
    presence_score: float
    motion_energy: float
    transition_density: float
    steady_pose_ratio: float
    lip_sync_status: str = "SKIPPED"
    black_frame_ratio: float = 0.0
    duration_delta_ms: float = 0.0


def benchmark_motion_quality(
    *,
    motion_path: Path,
    video_path: Path | None = None,
    audio_path: Path | None = None,
) -> MotionBenchmarkReport:
    track = PoseGraphTrack.from_npz(motion_path)
    result: MotionBenchmarkResult = benchmark_pose_track(track)

    report = MotionBenchmarkReport(
        gesture_variety=result.gesture_variety,
        presence_score=result.presence_score,
        motion_energy=result.motion_energy,
        transition_density=result.transition_density,
        steady_pose_ratio=result.steady_pose_ratio,
    )

    if video_path is not None and audio_path is not None and video_path.is_file() and audio_path.is_file():
        checker = VideoQualityChecker()
        qc = checker.check_quality(QCRequest(video_path=video_path, audio_path=audio_path))
        report.lip_sync_status = qc.status
        report.black_frame_ratio = qc.black_frame_ratio
        report.duration_delta_ms = qc.duration_delta_ms

    return report


def _write_result(
    path: Path,
    result: MotionBenchmarkReport,
    motion_path: Path,
    video_path: Path | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "motion_path": str(motion_path),
        "video_path": str(video_path) if video_path else None,
        "gesture_variety": result.gesture_variety,
        "presence_score": result.presence_score,
        "motion_energy": result.motion_energy,
        "transition_density": result.transition_density,
        "steady_pose_ratio": result.steady_pose_ratio,
        "lip_sync_status": result.lip_sync_status,
        "black_frame_ratio": result.black_frame_ratio,
        "duration_delta_ms": result.duration_delta_ms,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark motion presence and quality from extracted assets.")
    parser.add_argument("--motion", required=True, type=Path, help="hand_motion.npz or body_and_hands_motion.npz")
    parser.add_argument("--video", type=Path, default=None, help="Optional rendered video for lip-sync QC")
    parser.add_argument("--audio", type=Path, default=None, help="Optional audio for QC")
    parser.add_argument("--output", type=Path, default=Path("motion_benchmark.json"), help="JSON report path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = benchmark_motion_quality(motion_path=args.motion, video_path=args.video, audio_path=args.audio)
    _write_result(args.output, result, args.motion, args.video)
    print(json.dumps(
        {
            "gesture_variety": result.gesture_variety,
            "presence_score": result.presence_score,
            "motion_energy": result.motion_energy,
            "transition_density": result.transition_density,
            "steady_pose_ratio": result.steady_pose_ratio,
            "lip_sync_status": result.lip_sync_status,
            "black_frame_ratio": result.black_frame_ratio,
            "duration_delta_ms": result.duration_delta_ms,
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
