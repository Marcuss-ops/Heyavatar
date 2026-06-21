import os
import sys
import time
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
from workers.encoding_worker.worker import EncodingWorker

def main():
    os.environ["HEYAVATAR_MOCK_ENGINE"] = "0"
    get_settings.cache_clear()
    
    avatar_image = Path("C:/Users/pater/Pyt/Heyavatar/captures/Avatarpng/ChatGPT Image 27 mag 2026, 14_27_14 (1).png")
    
    # We can use captures/modulated_speech.wav if it exists, otherwise fallback to the test tone
    audio_path = Path("captures/modulated_speech.wav")
    if not audio_path.is_file():
        audio_path = Path("tmp_debug_test/speech.wav")
        if not audio_path.is_file():
            from tests.smoke.test_real_gpu._helpers import _test_audio
            audio_path = _test_audio(Path("tmp_debug_test"))
            
    print(f"Using avatar image: {avatar_image}")
    print(f"Using audio: {audio_path}")
    
    engine = get_provider(EngineId.LIVE_PORTRAIT)
    engine.load()
    # Enable static head logic so only facial features animate
    engine.inf_cfg.extra["static_head"] = True
    
    tmp_path = Path("tmp_debug_test")
    tmp_path.mkdir(exist_ok=True)
    
    try:
        # 1. Compile Identity
        print("Compiling user avatar...")
        pack_repo = AvatarPackRepository(root=tmp_path / "packs")
        spec = IdentitySpec(source_image=avatar_image, display_name="User Avatar")
        compiler = AvatarCompiler(engine=engine, pack_root=pack_repo.root)
        identity_handle = compiler.compile(spec)
        print(f"Identity compiled: {identity_handle.identity_id}")
        
        # 2. Render Crop Video (face_region_only = True)
        print("\nRendering Cropped Video (face region only)...")
        job_id_crop = RenderJobId("user_avatar_crop")
        request_crop = RenderRequest(
            job_id=job_id_crop,
            identity_id=identity_handle.identity_id,
            identity_spec=IdentitySpec(source_image=avatar_image),
            render_spec=RenderSpec(
                audio_path=audio_path, fps=25, target_resolution=(512, 512),
                face_region_only=True
            ),
            tier=Tier.EXPRESS,
        )
        
        rv = RenderVideo(
            engine=engine,
            chunk_config=ChunkConfig(chunk_seconds=4.0, overlap_seconds=0.0),
        )
        result_crop = rv.run(request_crop, identity_handle)
        
        encoder = EncodingWorker(settings=get_settings())
        final_crop_path = encoder.encode(
            str(job_id_crop),
            result_crop.output_path,
            audio_path=audio_path,
        )
        print(f"-> Saved cropped video to: {final_crop_path}")
        
        # 3. Render 1080p Video (face_region_only = False, static head)
        print("\nRendering Full 1080p Static Head Video...")
        job_id_full = RenderJobId("user_avatar_1080p_static")
        request_full = RenderRequest(
            job_id=job_id_full,
            identity_id=identity_handle.identity_id,
            identity_spec=IdentitySpec(source_image=avatar_image),
            render_spec=RenderSpec(
                audio_path=audio_path, fps=25, target_resolution=(1920, 1080),
                face_region_only=False
            ),
            tier=Tier.EXPRESS,
        )
        
        result_full = rv.run(request_full, identity_handle)
        final_full_path = encoder.encode(
            str(job_id_full),
            result_full.output_path,
            audio_path=audio_path,
        )
        print(f"-> Saved static head video to: {final_full_path}")
        
    finally:
        engine.unload()

if __name__ == "__main__":
    main()
