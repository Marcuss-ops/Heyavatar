import os
import sys
import time
from pathlib import Path
import torch
import numpy as np
import psutil

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

def get_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024) # MB

def main():
    os.environ["HEYAVATAR_MOCK_ENGINE"] = "0"
    get_settings.cache_clear()
    
    tmp_path = Path("tmp_debug_test")
    tmp_path.mkdir(exist_ok=True)
    
    # Clean previous runs
    for f in tmp_path.glob("*"):
        if f.is_file():
            f.unlink()
            
    from tests.smoke.test_real_gpu._helpers import _test_image, _test_audio
    source = _test_image(tmp_path)
    audio = _test_audio(tmp_path)
    
    print("=== RESOURCE USAGE METRICS FOR HEYAVATAR VIDEO GENERATION ===")
    
    # Baseline
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_ram = get_ram_usage()
    base_vram = torch.cuda.memory_allocated() / (1024 * 1024) # MB
    print(f"Baseline System RAM: {base_ram:.2f} MB")
    print(f"Baseline GPU VRAM: {base_vram:.2f} MB")
    
    # 1. Load Engine
    t0 = time.perf_counter()
    engine = get_provider(EngineId.LIVE_PORTRAIT)
    engine.load()
    load_time = time.perf_counter() - t0
    
    post_load_ram = get_ram_usage()
    post_load_vram = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"\n[Engine Load]")
    print(f"Load duration: {load_time:.2f} seconds")
    print(f"RAM used (post-load): {post_load_ram - base_ram:.2f} MB (Total: {post_load_ram:.2f} MB)")
    print(f"VRAM allocated (post-load): {post_load_vram - base_vram:.2f} MB (Total: {post_load_vram:.2f} MB)")
    
    try:
        # 2. Compile Identity
        t0 = time.perf_counter()
        pack_repo = AvatarPackRepository(root=tmp_path / "packs")
        spec = IdentitySpec(source_image=source, display_name="Actor 1")
        compiler = AvatarCompiler(engine=engine, pack_root=pack_repo.root)
        identity_handle = compiler.compile(spec)
        compile_time = time.perf_counter() - t0
        
        post_compile_ram = get_ram_usage()
        post_compile_vram = torch.cuda.memory_allocated() / (1024 * 1024)
        print(f"\n[Identity Compilation]")
        print(f"Compile duration: {compile_time:.2f} seconds")
        print(f"RAM used (post-compile increase): {post_compile_ram - post_load_ram:.2f} MB")
        print(f"VRAM allocated (post-compile increase): {post_compile_vram - post_load_vram:.2f} MB")
        
        # 3. Render Chunks
        t0 = time.perf_counter()
        job_id = RenderJobId("job-measure")
        request = RenderRequest(
            job_id=job_id,
            identity_id=identity_handle.identity_id,
            identity_spec=IdentitySpec(source_image=source),
            render_spec=RenderSpec(
                audio_path=audio, fps=25, target_resolution=(512, 512),
                face_region_only=True
            ),
            tier=Tier.EXPRESS,
        )
        
        rv = RenderVideo(
            engine=engine,
            telemetry=TelemetryRecorder(),
            chunk_config=ChunkConfig(chunk_seconds=2.0, overlap_seconds=0.0),
        )
        torch.cuda.reset_peak_memory_stats()
        result = rv.run(request, identity_handle)
        render_time = time.perf_counter() - t0
        
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
        post_render_ram = get_ram_usage()
        post_render_vram = torch.cuda.memory_allocated() / (1024 * 1024)
        
        print(f"\n[Video Rendering (1.0s video, 25 frames, 256x256 crop)]")
        print(f"Render duration: {render_time:.2f} seconds")
        print(f"FPS achieved: {25 / render_time:.2f} frames/sec")
        print(f"Telemetry GPU time: {result.gpu_seconds_total:.2f} seconds")
        print(f"RAM used (post-render increase): {post_render_ram - post_compile_ram:.2f} MB")
        print(f"Peak VRAM during rendering: {peak_vram:.2f} MB")
        print(f"Final VRAM allocated: {post_render_vram:.2f} MB")
        
        # 4. Encoding
        t0 = time.perf_counter()
        encoder = EncodingWorker(settings=get_settings())
        final_path = encoder.encode(
            str(job_id),
            result.output_path,
            audio_path=audio,
        )
        encode_time = time.perf_counter() - t0
        print(f"\n[Video Encoding (FFmpeg muxing)]")
        print(f"Encode duration: {encode_time:.2f} seconds")
        print(f"Final output size: {final_path.stat().st_size / 1024:.2f} KB")
        
    finally:
        engine.unload()
        print("\n[Engine Unloaded]")
        torch.cuda.empty_cache()
        print(f"Final RAM: {get_ram_usage():.2f} MB")
        print(f"Final VRAM: {torch.cuda.memory_allocated() / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    main()
