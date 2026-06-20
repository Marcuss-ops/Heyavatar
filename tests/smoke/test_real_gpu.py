"""Real-GPU smoke test for the first production video.

This test verifies the full pipeline with ``HEYAVATAR_MOCK_ENGINE=0``
and a real NVIDIA GPU. It is intended to be run on a dedicated GPU
worker node (with CUDA toolkit, MSVC/Linux build tools, and the
LivePortrait upstream repo cloned and built).

Pre-requisites (before running this test):
1. ``pip install torch --index-url https://download.pytorch.org/whl/cu124``
2. ``pip install huggingface_hub``
3. Clone LivePortrait: ``git clone https://github.com/KlingAIResearch/LivePortrait``
4. Build the CUDA op: ``cd LivePortrait && bash tools/prepare_env.sh``
   (on Windows, install MSVC Build Tools first)
5. Add the repo root to PYTHONPATH (so ``src.live_portrait_pipeline`` resolves):
   ``set PYTHONPATH=%CD%\LivePortrait;%PYTHONPATH%``
6. Set env vars:
   ``set HEYAVATAR_MOCK_ENGINE=0``
   ``set HEYAVATAR_SKIP_SHA256_VERIFY=1``
   ``set HEYAVATAR_LIVE_PORTRAIT_SRC=%CD%\LivePortrait``
7. Run: ``pytest tests/smoke/test_real_gpu.py -v -s``

What this test verifies
-----------------------
- GPU + CUDA are available and healthy
- All 5 LivePortrait checkpoint files exist and are non-empty
- Checkpoint hash verification works (or SKIP_SHA256_VERIFY flag)
- The engine loads successfully in real mode
- ``prepare_identity()`` produces real source features (non-mock)
- ``render_chunk()`` produces a valid mp4
- The full pipeline (compile → render → encode) runs end-to-end
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from contracts.job_queue import JobState, QueueHandle, RenderJob
from src.core.config import get_settings
from src.domain.enums import EngineId, Tier
from src.domain.types import (
    IdentitySpec,
    IdentityId,
    RenderJobId,
    RenderRequest,
    RenderSpec,
)
from src.scheduler.queue import InMemoryJobQueue
from src.storage.avatar_packs import AvatarPackRepository
from src.storage.jobs import InMemoryJobRepository
from providers.liveportrait.checkpoint_manager import CHECKPOINT_MANIFEST
from tests._fixtures import PNG_1X1 as _PNG_1x1


# ── guards ──────────────────────────────────────────────────────────


requires_cuda = pytest.mark.skipif(
    _reason := "CUDA not available",
)
try:
    import torch  # type: ignore
    if not torch.cuda.is_available():
        requires_cuda = pytest.mark.skipif(True, reason="CUDA not available")
    else:
        requires_cuda = pytest.mark.skipif(False, reason="")
except ImportError:
    requires_cuda = pytest.mark.skipif(True, reason="PyTorch not installed")


# ── path setup ─────────────────────────────────────────────────────

# LivePortrait uses relative imports (``from .config ...``) inside
# ``src/``, so the *repo root* (parent of ``src/``) must be on
# ``sys.path`` so ``import src.live_portrait_pipeline`` resolves.
_LIVE_PORTRAIT_REPO = Path(__file__).resolve().parents[2] / "LivePortrait"
if str(_LIVE_PORTRAIT_REPO) not in os.environ.get("PYTHONPATH", ""):
    existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = (
        f"{_LIVE_PORTRAIT_REPO}{os.pathsep}{existing}" if existing else str(_LIVE_PORTRAIT_REPO)
    )
if str(_LIVE_PORTRAIT_REPO) not in sys.path:
    sys.path.insert(0, str(_LIVE_PORTRAIT_REPO))

# Also register the repo root so the adapter's _import_upstream_live_portrait
# can find it via HEYAVATAR_LIVE_PORTRAIT_SRC.
os.environ.setdefault("HEYAVATAR_LIVE_PORTRAIT_SRC", str(_LIVE_PORTRAIT_REPO))


# ── helpers ─────────────────────────────────────────────────────────


def _test_image(tmp_path: Path) -> Path:
    """Write a real-ish 1x1 PNG for identity preparation."""
    p = tmp_path / "actor.png"
    p.write_bytes(_PNG_1x1)
    return p


def _test_audio(tmp_path: Path) -> Path:
    """Write a minimal valid WAV for render_chunk."""
    p = tmp_path / "speech.wav"
    # Minimal WAV: 44-byte header + 100 bytes of silence.
    import struct
    sample_rate = 16000
    num_samples = 50
    data = b"\x00" * (num_samples * 2)  # 16-bit mono silence
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(data),
        b"WAVE",
        b"fmt ", 16, 1, 1, sample_rate,
        sample_rate * 2, 2, 16,
        b"data", len(data),
    )
    p.write_bytes(header + data)
    return p


# ── tests ───────────────────────────────────────────────────────────


@requires_cuda
def test_gpu_cuda_healthy():
    """Verify the GPU is reachable and has sufficient VRAM."""
    assert torch.cuda.is_available(), "CUDA must be available"
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    vram_gb = props.total_memory // (1 << 30)
    assert vram_gb >= 4, f"Need >= 4 GB VRAM, got {vram_gb} GB"
    # Smoke: allocate a small tensor and run an op.
    x = torch.randn(100, 100, device=device)
    y = x @ x.T
    assert y.shape == (100, 100)
    print(f"\nGPU: {props.name} ({vram_gb} GB VRAM)")


@requires_cuda
def test_checkpoints_present_and_readable():
    """Verify all 5 LivePortrait checkpoint files exist and are non-empty."""
    root = Path(os.environ.get(
        "HEYAVATAR_LIVE_PORTRAIT_CHECKPOINTS",
        "./checkpoints/liveportrait",
    ))
    assert root.is_dir(), f"Checkpoint root missing: {root}"

    total_mb = 0
    for entry in CHECKPOINT_MANIFEST:
        name = entry["name"]
        path = root / name
        assert path.is_file(), f"Missing checkpoint: {name} at {path}"
        size = path.stat().st_size
        assert size > 0, f"Checkpoint {name} is empty"
        total_mb += size
        print(f"  {name}: {size / (1<<20):.1f} MB")
    print(f"  Total: {total_mb / (1<<20):.0f} MB")


@requires_cuda
def test_checkpoint_sha256_verification():
    """Compute SHA256 of each checkpoint file for pinning."""
    root = Path(os.environ.get(
        "HEYAVATAR_LIVE_PORTRAIT_CHECKPOINTS",
        "./checkpoints/liveportrait",
    ))
    print("\nSHA256 values (copy to checkpoint_manager.py):")
    for entry in CHECKPOINT_MANIFEST:
        name = entry["name"]
        path = root / name
        if not path.is_file():
            print(f"  {name}: MISSING")
            continue
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        print(f'  \"{name}\": \"{h.hexdigest()}\"')
    print("Set HEYAVATAR_SKIP_SHA256_VERIFY=1 until these are pinned.")


@requires_cuda
def test_engine_loads_in_real_mode(workdir, tmp_path):
    """Verify the LivePortrait engine loads successfully with GPU."""
    # Override the session-wide mock-engine fixture for real GPU tests.
    os.environ["HEYAVATAR_MOCK_ENGINE"] = "0"
    from src.core.config import get_settings
    get_settings.cache_clear()
    settings = get_settings()
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from providers import get_provider
    engine = get_provider(EngineId.LIVE_PORTRAIT)
    assert engine.engine_id == EngineId.LIVE_PORTRAIT

    try:
        engine.load()
        health = engine.health()
        print(f"\nEngine state after load: {health.state.value}")
        assert health.state.value in ("idle", "loading"), (
            f"Engine should be IDLE or LOADING after load(), got {health.state.value}"
        )
        assert health.mock_mode is False, "Should be in real mode"
        assert health.vram_used_mb >= 0
    finally:
        engine.unload()


@requires_cuda
def test_prepare_identity_real_mode(workdir, tmp_path):
    """Verify prepare_identity() produces real (non-mock) assets."""
    settings = get_settings()
    if settings.mock_engine:
        pytest.skip("HEYAVATAR_MOCK_ENGINE=1")

    source = _test_image(tmp_path)

    from providers import get_provider
    engine = get_provider(EngineId.LIVE_PORTRAIT)
    engine.load()
    try:
        assets = engine.prepare_identity(source)
        assert isinstance(assets, dict)
        # Real mode must produce the full feature volume (not mock bytes).
        assert "source_features.bin" in assets
        f_s_bytes = assets["source_features.bin"]
        assert len(f_s_bytes) > 1000, (
            f"source_features.bin too small ({len(f_s_bytes)} B) — "
            "likely mock mode or degraded"
        )
        # Should produce a real face crop (not the mock random noise).
        assert "face_crop.png" in assets
        # The canonical keypoints should be present.
        assert "canonical_keypoints.bin" in assets
        print(f"\nIdentity prepared: {len(assets)} assets, "
              f"source_features={len(f_s_bytes)} B")
    finally:
        engine.unload()


@requires_cuda
def test_full_pipeline_real_mode(workdir, tmp_path):
    """End-to-end: compile identity → render chunk → encode → mp4."""
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
        # ── 2. Compile identity ─────────────────────────────────
        from src.application.compile_avatar import AvatarCompiler
        from src.domain.avatar_pack import read_pack

        pack_repo = AvatarPackRepository(root=workdir / "packs")
        spec = IdentitySpec(source_image=source, display_name="Actor 1")
        compiler = AvatarCompiler(engine=engine, pack_root=pack_repo.root)
        identity_handle = compiler.compile(spec)
        pack_repo.save(identity_handle.identity_id, read_pack(identity_handle.pack_path))
        print(f"\nIdentity compiled: {identity_handle.identity_id}")

        # ── 3. Build render request ─────────────────────────────
        job_id = RenderJobId("job-real-gpu-001")
        request = RenderRequest(
            job_id=job_id,
            identity_id=identity_handle.identity_id,
            identity_spec=IdentitySpec(source_image=source),
            render_spec=RenderSpec(audio_path=audio, fps=25, target_resolution=(512, 512)),
            tier=Tier.EXPRESS,
        )

        # ── 4. Render chunks ────────────────────────────────────
        from src.application.render_video import ChunkConfig, RenderVideo
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
        print(f"Chunks: {len(result.chunks)}, GPU seconds: {result.gpu_seconds_total:.2f}")

        # ── 5. Encode final mp4 ─────────────────────────────────
        from workers.encoding_worker import EncodingWorker
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

        # ── 6. Verify telemetry ─────────────────────────────────
        assert rv.telemetry.gpu_seconds_total > 0, "Telemetry should record GPU time"
        print(f"GPU-seconds total: {rv.telemetry.gpu_seconds_total:.2f}")

    finally:
        engine.unload()
