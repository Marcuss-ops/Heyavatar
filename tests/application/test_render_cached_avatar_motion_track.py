from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from contracts.quality_checker import QCResult
from src.application import render_cached_avatar as rca
from src.domain.body_template import BodyTemplate
from src.domain.types import AvatarIdentityHandle


@dataclass
class _FakeChunk:
    output_path: Path
    duration_seconds: float = 1.0
    gpu_seconds: float = 0.1
    engine_id: object = None

    def __post_init__(self) -> None:
        if self.engine_id is None:
            self.engine_id = SimpleNamespace(value="musetalk-v1")


def test_render_cached_avatar_resolves_gesture_from_motion_track(tmp_path: Path, monkeypatch) -> None:
    motion_path = tmp_path / "hand_motion.npz"
    np.savez(
        motion_path,
        fps=np.asarray([25.0], dtype=np.float32),
        pose_state=np.asarray(["neutral_desk", "right_hand_up"], dtype="U32"),
        transition_frames=np.asarray([1], dtype=np.int32),
        hold_frames=np.asarray([1], dtype=np.int32),
        left_hand_landmarks=np.zeros((2, 21, 3), dtype=np.float32),
        right_hand_landmarks=np.zeros((2, 21, 3), dtype=np.float32),
        left_wrist_velocity=np.zeros(2, dtype=np.float32),
        right_wrist_velocity=np.zeros(2, dtype=np.float32),
    )

    body_dir = tmp_path / "avatar_packs" / "avatar-1" / "body_cache" / "explain_right"
    body_dir.mkdir(parents=True, exist_ok=True)
    for name in ["body.mp4", "face_mask.mp4", "neck_mask.mp4", "metadata.json"]:
        (body_dir / name).write_bytes(b"ok")
    np.savez(body_dir / "face_transforms.npz", bbox=np.array([[0, 0, 10, 10]], dtype=np.float32))

    captured = {}

    def _loader(avatar_id: str, gesture_id: str, *, base_dir: str | Path = "avatar_packs") -> BodyTemplate:
        captured["gesture_id"] = gesture_id
        return BodyTemplate(
            body_video=body_dir / "body.mp4",
            face_mask=body_dir / "face_mask.mp4",
            neck_mask=body_dir / "neck_mask.mp4",
            face_transforms=body_dir / "face_transforms.npz",
            metadata=body_dir / "metadata.json",
        )

    fake_handle = AvatarIdentityHandle(
        identity_id="id-1",
        pack_path=tmp_path / "pack.tar",
        pack_digest="digest",
        prepared_at=SimpleNamespace(),
    )

    class _FakeEngine:
        def __init__(self) -> None:
            self.settings = SimpleNamespace(capture_dir=tmp_path / "captures", pack_dir=tmp_path / "packs")

        def render_chunk(self, request, identity):
            out = self.settings.capture_dir / request.job_id / "chunk_0000.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"chunk")
            return _FakeChunk(output_path=out)

    monkeypatch.setattr(rca, "_resolve_identity", lambda **_k: (fake_handle, True, True))
    monkeypatch.setattr(rca, "extract_face_roi", lambda *args, **kwargs: None)
    monkeypatch.setattr(rca, "mux_audio", lambda *args, **kwargs: Path(args[2]).write_bytes(b"final"))

    class _FakeCompositor:
        def composite(self, request):
            request.output_path.write_bytes(b"composited")
            return SimpleNamespace(frames_processed=2)

    monkeypatch.setattr(rca, "OpenCVFaceCompositor", lambda: _FakeCompositor())
    monkeypatch.setattr(
        rca,
        "VideoQualityChecker",
        lambda: SimpleNamespace(
            check_quality=lambda *_a, **_k: QCResult(
                passed=True,
                status="COMPLETED",
                debug_green_ratio=0.0,
                black_frame_ratio=0.0,
                duration_delta_ms=0.0,
                frames_expected=2,
                frames_actual=2,
                invalid_transforms=0,
            )
        ),
    )

    result = rca.render_cached_avatar(
        avatar_id="avatar-1",
        gesture_id="auto",
        identity_id="id-1",
        audio_path=tmp_path / "speech.wav",
        output_path=tmp_path / "final.mp4",
        engine=_FakeEngine(),
        body_template_loader=_loader,
        body_templates_dir=tmp_path / "avatar_packs",
        motion_track_path=motion_path,
    )

    assert captured["gesture_id"] == "explain_right"
    assert result.metrics["motion_track"]["unique_states"] == ["right_hand_up"]
    assert result.metrics["motion_benchmark"]["presence_score"] >= 0.0
