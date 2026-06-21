import os
import sys
from pathlib import Path
import torch
import numpy as np

# Setup python path to include LivePortrait
liveportrait_path = Path("LivePortrait").resolve()
sys.path.insert(0, str(liveportrait_path))
sys.path.insert(0, str(Path("src").resolve()))
os.environ["HEYAVATAR_LIVE_PORTRAIT_SRC"] = str(liveportrait_path)

from tests.smoke.test_real_gpu._helpers import _setup_live_portrait_path
_setup_live_portrait_path()

from providers import get_provider
from src.domain.enums import EngineId, Tier
from src.core.config import get_settings
from src.domain.types import IdentitySpec, RenderJobId, RenderRequest, RenderSpec
from src.application.compile_avatar import AvatarCompiler
from src.storage.avatar_packs import AvatarPackRepository
from src.application.render_video.use_case import RenderVideo
from src.application.render_video.config import ChunkConfig
from src.application.telemetry import TelemetryRecorder
from workers.encoding_worker.worker import EncodingWorker

def main():
    os.environ["HEYAVATAR_MOCK_ENGINE"] = "0"
    get_settings.cache_clear()
    
    tmp_path = Path("tmp_debug_test")
    tmp_path.mkdir(exist_ok=True)
    
    # 1. Clean previous runs
    for f in tmp_path.glob("*"):
        if f.is_file():
            f.unlink()
            
    from tests.smoke.test_real_gpu._helpers import _test_image, _test_audio
    source = _test_image(tmp_path)
    audio = _test_audio(tmp_path)
    
    engine = get_provider(EngineId.LIVE_PORTRAIT)
    engine.load()
    
    try:
        pack_repo = AvatarPackRepository(root=tmp_path / "packs")
        spec = IdentitySpec(source_image=source, display_name="Actor 1")
        compiler = AvatarCompiler(engine=engine, pack_root=pack_repo.root)
        identity_handle = compiler.compile(spec)
        
        job_id = RenderJobId("job-debug-flow")
        request = RenderRequest(
            job_id=job_id,
            identity_id=identity_handle.identity_id,
            identity_spec=IdentitySpec(source_image=source),
            render_spec=RenderSpec(
                audio_path=audio, fps=25, target_resolution=(512, 512),
                face_region_only=False
            ),
            tier=Tier.EXPRESS,
        )
        
        rv = RenderVideo(
            engine=engine,
            telemetry=TelemetryRecorder(),
            chunk_config=ChunkConfig(chunk_seconds=2.0, overlap_seconds=0.0),
        )
        result = rv.run(request, identity_handle)
        print("Wrote manifest to:", result.output_path)
        
        # Print manifest contents
        print("Manifest content:")
        print(result.output_path.read_text())
        
        encoder = EncodingWorker(settings=get_settings())
        final_path = encoder.encode(
            str(job_id),
            result.output_path,
            audio_path=audio,
        )
        print("Encoded final mp4:", final_path)
        
        # Read frames of final mp4
        from tests.smoke.test_real_gpu._helpers import _read_mp4_frames
        frames = _read_mp4_frames(final_path)
        print("Frames count:", len(frames))
        
        def _lower_face_ssd(prev, curr) -> float:
            h = prev.shape[0]
            y0 = (2 * h) // 3
            a = prev[y0:, :, :].astype(np.float32)
            b = curr[y0:, :, :].astype(np.float32)
            return float(np.mean((a - b) ** 2))
            
        mid = len(frames) // 2
        for i in range(len(frames) - 1):
            ssd = _lower_face_ssd(frames[i], frames[i + 1])
            print(f"Frame {i} -> {i+1} SSD: {ssd:.6f}")
            
    finally:
        engine.unload()

if __name__ == "__main__":
    main()
