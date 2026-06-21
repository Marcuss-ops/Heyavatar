"""End-to-end real-mode pipeline: compile → render → encode → mp4.

The full happy path on a real GPU worker, asserting no degraded
chunks, measurable GPU time, and a valid encoded mp4 on disk.
"""

from __future__ import annotations

import pytest

from src.core.config import get_settings
from src.domain.avatar_pack import read_pack
from src.domain.enums import EngineId, Tier
from src.domain.types import (
    IdentitySpec,
    RenderJobId,
    RenderRequest,
    RenderSpec,
)
from src.storage.avatar_packs import AvatarPackRepository
from tests.smoke.test_real_gpu._helpers import (
    _test_audio,
    _test_image,
    real_mode_env,  # noqa: F401, F811 (pytest fixture — ruff can't see fixture lookup)
    requires_cuda,
    requires_ffmpeg,
)


@requires_cuda
@requires_ffmpeg
def test_full_pipeline_real_mode(real_mode_env, workdir, tmp_path):  # noqa: F811 (pytest fixture — ruff can't see fixture lookup)
    """End-to-end: compile identity → render chunk → encode → mp4.

    Validates two release-critical invariants:

    1. No degraded chunks (the renderer actually ran on GPU).
    2. The produced mp4 has per-frame motion in the active-speech
       window of the audio fixture (mouth drives the avatar in
       time with the audio, NOT the static `0x111111` black fallback
       or the `0x330000` degraded fallback).
    """
    settings = get_settings()
    if settings.mock_engine:
        pytest.skip("HEYAVATAR_MOCK_ENGINE=1")

    source = _test_image(tmp_path)
    audio = _test_audio(tmp_path)

    # ── 1. Engine setup ─────────────────────────────────────────
    from providers import get_provider

    engine = get_provider(EngineId.LIVE_PORTRAIT)
    engine.load()
    try:
        # ── 2. Compile identity ──────────────────────────────────
        from src.application.compile_avatar import AvatarCompiler

        pack_repo = AvatarPackRepository(root=workdir / "packs")
        spec = IdentitySpec(source_image=source, display_name="Actor 1")
        compiler = AvatarCompiler(engine=engine, pack_root=pack_repo.root)
        identity_handle = compiler.compile(spec)
        pack_repo.save(identity_handle.identity_id, read_pack(identity_handle.pack_path))
        print(f"\nIdentity compiled: {identity_handle.identity_id}")

        # ── 3. Build render request ──────────────────────────────
        job_id = RenderJobId("job-real-gpu-001")
        request = RenderRequest(
            job_id=job_id,
            identity_id=identity_handle.identity_id,
            identity_spec=IdentitySpec(source_image=source),
            render_spec=RenderSpec(
                audio_path=audio, fps=25, target_resolution=(512, 512)
            ),
            tier=Tier.EXPRESS,
        )

        # ── 4. Render chunks ─────────────────────────────────────
        from src.application.render_video.config import ChunkConfig
        from src.application.render_video.use_case import RenderVideo
        from src.application.telemetry import TelemetryRecorder

        rv = RenderVideo(
            engine=engine,
            telemetry=TelemetryRecorder(),
            chunk_config=ChunkConfig(chunk_seconds=2.0, overlap_seconds=0.0),
        )
        result = rv.run(request, identity_handle)
        assert len(result.chunks) >= 1, "Should produce at least 1 chunk"
        assert result.gpu_seconds_total > 0, "Should measure real GPU time"
        assert len(result.degraded_chunks) == 0, (
            f"Should have zero degraded chunks, got {result.degraded_chunks}"
        )
        print(
            f"Chunks: {len(result.chunks)}, GPU seconds: {result.gpu_seconds_total:.2f}"
        )

        # ── 5. Encode final mp4 ──────────────────────────────────
        from workers.encoding_worker.worker import EncodingWorker

        encoder = EncodingWorker(settings=settings)
        manifest_path = result.output_path
        assert manifest_path.is_file(), f"Manifest missing: {manifest_path}"
        final_path = encoder.encode(
            str(job_id),
            manifest_path,
            audio_path=audio,
        )
        assert final_path.is_file(), f"Final mp4 missing: {final_path}"
        assert final_path.stat().st_size > 0, "Final mp4 should be non-empty"
        print(f"Final mp4: {final_path} ({final_path.stat().st_size} bytes)")

        # ── 6. Verify telemetry ──────────────────────────────────
        assert rv.telemetry.gpu_seconds_total > 0, "Telemetry should record GPU time"
        print(f"GPU-seconds total: {rv.telemetry.gpu_seconds_total:.2f}")

        # ── 7. Mouth-sync assertion: per-frame SSD in the lower
        # face region. The audio fixture is 0.5s silence + 0.5s of
        # an active 1 kHz tone; we partition the decoded frames the
        # same way (by frame index, since the video rate matches the
        # audio duration). SSD_silence should be near zero (the
        # avatar is essentially still); SSD_speech should be clearly
        # larger because the neural bridge drives the mouth from the
        # active audio. If both are near zero, the renderer
        # silently fell back to the `0x111111`/`0x330000` dummy and
        # we're shipping a broken artefact to paying customers.
        from tests.smoke.test_real_gpu._helpers import _read_mp4_frames

        frames = _read_mp4_frames(final_path)
        assert frames, "decoded mp4 yielded zero frames"
        # ``_test_audio`` is 1.0s of audio, rendered at 25 fps → 25
        # frames; split into the first 12 frames (silence) and last
        # 13 frames (active tone). Take the lower 1/3 of the frame
        # as the mouth-region proxy — in LivePortrait's 512x512
        # output the mouth sits roughly in y ∈ [2*H/3, H].
        import numpy as _np

        def _lower_face_ssd(prev, curr) -> float:
            h = prev.shape[0]
            y0 = (2 * h) // 3
            a = prev[y0:, :, :].astype(_np.float32)
            b = curr[y0:, :, :].astype(_np.float32)
            return float(_np.mean((a - b) ** 2))

        # Partition into silence (first half) and speech (second
        # half); the 1-second fixture at 25 fps splits cleanly.
        mid = len(frames) // 2
        ssd_silence = [
            _lower_face_ssd(frames[i], frames[i + 1])
            for i in range(mid - 1)
        ]
        ssd_speech = [
            _lower_face_ssd(frames[i], frames[i + 1])
            for i in range(mid, len(frames) - 1)
        ]
        med_silence = (
            float(_np.median(ssd_silence)) if ssd_silence else 0.0
        )
        med_speech = float(_np.median(ssd_speech)) if ssd_speech else 0.0
        # Threshold rationale: a real LP-7 mouth opening across a
        # 512-row × 256-col mouth-ROI region at 8-bit RGB produces an
        # SSD in the hundreds-to-thousands due to per-pixel motion
        # deltas. The 0.5s silence window should have SSD near zero
        # because the renderer should hold the avatar pose steady.
        # 200.0 distinguishes real motion from numerical micro-jitter
        # on a static red avatar without being permissive enough to
        # let a degraded 0x330000 frame slip through.
        assert med_speech > 200.0, (
            f"Median SSD in the active-speech window ({med_speech:.2f}) "
            "is too low to represent real lip motion. The renderer "
            "likely fell back to the static black/dummy output. "
            "Fix the audio bridge before shipping."
        )
        # Silence baseline: SSD_silence should be near zero on a
        # correctly-held pose. The ``< 10.0`` threshold rejects both
        # the inverse bug (mouth moves when there's no audio) and a
        # noisy jitter between consecutive frames in the silence
        # window — a static frame-to-frame SSD on a red avatar is
        # exactly 0 modulo the small static sway in the head-pose
        # micro-movement code path (a few px of head motion, well
        # below 10 SSD units).
        assert med_silence < 10.0, (
            f"Median SSD in the silence window ({med_silence:.2f}) is "
            f"unexpectedly high — the renderer should hold the avatar "
            f"pose steady when there's no audio. Got SSD_silence="
            f"{med_silence:.2f}, expected < 10."
        )
        # And the inverse bug catcher: mouth moves on silence, not
        # speech.
        assert med_speech > med_silence, (
            f"Median SSD_speech ({med_speech:.2f}) is not greater than "
            f"Median SSD_silence ({med_silence:.2f}). The neural bridge "
            f"is driving the mouth on silence, not speech."
        )
        print(
            f"Mouth-sync OK: SSD_silence={med_silence:.2f}, "
            f"SSD_speech={med_speech:.2f}"
        )

    finally:
        engine.unload()
