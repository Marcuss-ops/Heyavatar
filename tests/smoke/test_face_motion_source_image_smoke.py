"""Smoke test: source_image -> compile -> render -> encode with face motion."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from providers import get_provider
from providers._ffmpeg import _write_dummy_mp4
from src.application.run_cached_avatar_from_text import run_cached_avatar_from_text
from src.domain.enums import EngineId
from tests.smoke.test_real_gpu._helpers import _test_audio, _test_image


requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="mock-mode smoke test shells out to ffmpeg",
)


def _write_idle_body_template(base: Path, avatar_id: str) -> None:
    pack = base / avatar_id / "body_cache" / "idle_small"
    pack.mkdir(parents=True, exist_ok=True)
    _write_dummy_mp4(pack / "body.mp4", duration=1.0, fps=25, resolution=(512, 512))
    _write_dummy_mp4(pack / "face_mask.mp4", duration=1.0, fps=25, colour="0x222222", resolution=(512, 512))
    _write_dummy_mp4(pack / "neck_mask.mp4", duration=1.0, fps=25, colour="0x111111", resolution=(512, 512))
    import numpy as np

    np.savez_compressed(
        pack / "face_transforms.npz",
        bbox=np.array([[0, 0, 512, 512]], dtype=np.float32),
        affine=np.eye(3, dtype=np.float32),
    )
    (pack / "metadata.json").write_text('{"pose_id":"neutral_desk"}', encoding="utf-8")


@requires_ffmpeg
def test_face_motion_smoke_source_image_compile_render_encode(workdir, tmp_path):
    source = _test_image(tmp_path)
    audio = _test_audio(tmp_path)
    avatar_id = "smoke_actor"
    _write_idle_body_template(workdir, avatar_id)

    engine = get_provider(EngineId.MUSE_TALK)
    engine.load()
    try:
        result = run_cached_avatar_from_text(
            text="Ciao a tutti, oggi vediamo tre differenze molto importanti?",
            audio_path=audio,
            output_path=tmp_path / "final.mp4",
            avatar_id=avatar_id,
            engine=engine,
            language="it",
            source_image=source,
            body_templates_dir=workdir,
            capture_root=tmp_path / "captures",
            mode="timeline",
        )

        assert result.output_path.is_file()
        assert result.face_timeline_path.is_file()
        assert result.face_motion_profile["segment_count"] >= 1
        assert result.face_motion_profile["motion_ids"]
    finally:
        engine.unload()
