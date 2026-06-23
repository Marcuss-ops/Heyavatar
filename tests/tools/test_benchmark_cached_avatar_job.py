from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.avatar_assets.benchmark_cached_avatar_job import benchmark_cached_avatar_job, main


def test_benchmark_cached_avatar_job_reads_runtime_dir(tmp_path: Path, monkeypatch) -> None:
    runtime_dir = tmp_path / "captures" / "job-1"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        runtime_dir / "motion.npz",
        fps=np.asarray([25.0], dtype=np.float32),
        pose_state=np.asarray(["neutral_desk", "right_hand_up"], dtype="U32"),
        transition_frames=np.asarray([1], dtype=np.int32),
        hold_frames=np.asarray([1], dtype=np.int32),
        left_hand_landmarks=np.zeros((2, 21, 3), dtype=np.float32),
        right_hand_landmarks=np.zeros((2, 21, 3), dtype=np.float32),
        left_wrist_velocity=np.zeros(2, dtype=np.float32),
        right_wrist_velocity=np.array([0.0, 0.2], dtype=np.float32),
    )
    (runtime_dir / "metrics.json").write_text(
        json.dumps(
            {
                "motion_track_path": str(runtime_dir / "motion.npz"),
                "gpu_seconds": 12.0,
                "wall_seconds": 6.0,
                "gpu_seconds_per_output_minute": 120.0,
            }
        ),
        encoding="utf-8",
    )
    (runtime_dir / "result.json").write_text(json.dumps({"status": "COMPLETED", "final_path": str(runtime_dir / "final.mp4")}), encoding="utf-8")
    (runtime_dir / "final.mp4").write_bytes(b"fake")

    class _FakeQC:
        status = "COMPLETED"
        debug_green_ratio = 0.0
        black_frame_ratio = 0.0
        duration_delta_ms = 0.0

    monkeypatch.setattr("tools.avatar_assets.benchmark_cached_avatar_job.VideoQualityChecker", lambda: type("_Checker", (), {"check_quality": lambda self, *_a, **_k: _FakeQC()})())

    report = benchmark_cached_avatar_job(runtime_dir, audio_path=None)

    assert report.status == "COMPLETED"
    assert report.presence_score > 0.0
    assert report.gpu_seconds == 12.0


def test_benchmark_cached_avatar_job_cli(tmp_path: Path, monkeypatch) -> None:
    runtime_dir = tmp_path / "captures" / "job-2"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "metrics.json").write_text("{}", encoding="utf-8")
    (runtime_dir / "result.json").write_text("{}", encoding="utf-8")
    (runtime_dir / "final.mp4").write_bytes(b"fake")
    monkeypatch.setattr(
        "tools.avatar_assets.benchmark_cached_avatar_job.benchmark_cached_avatar_job",
        lambda runtime_dir, audio_path=None: type(
            "_Report",
            (),
            {
                "status": "UNKNOWN",
                "final_path": str(runtime_dir / "final.mp4"),
                "motion_track_path": "",
                "gesture_variety": 0.0,
                "presence_score": 0.0,
                "motion_energy": 0.0,
                "transition_density": 0.0,
                "steady_pose_ratio": 1.0,
                "qc_status": "SKIPPED",
                "debug_green_ratio": 0.0,
                "black_frame_ratio": 0.0,
                "duration_delta_ms": 0.0,
                "gpu_seconds": 0.0,
                "wall_seconds": 0.0,
                "gpu_seconds_per_output_minute": 0.0,
            },
        )(),
    )

    exit_code = main(["--runtime-dir", str(runtime_dir), "--output", str(tmp_path / "report.json")])

    assert exit_code == 0
    assert (tmp_path / "report.json").is_file()
