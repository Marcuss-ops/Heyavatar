import sys
import os
import time
from pathlib import Path
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.application.render_video.audio_probe import _probe_audio_duration
from src.application.render_video.face_region import mux_audio
from providers import get_provider
from src.domain.enums import EngineId
from src.domain.types import RenderChunkRequest, AvatarIdentityHandle, IdentityId

def main():
    source_img_path = Path(r"C:\Users\pater\Downloads\ChatGPT Image 27 mag 2026, 14_27_14 (4).png")
    audio_path = Path("bench_run/speech_edge.wav")
    output_path = Path("captures/static_animated.mp4")
    temp_output_path = Path("captures/static_animated_temp.mp4")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Rename TV studio background to avoid chroma keying the user's original image
    bg_path = Path("assets/tv_studio_background.png")
    bg_backup = Path("assets/tv_studio_background_backup.png")
    if bg_path.exists():
        print("Backing up tv_studio_background.png...")
        bg_path.rename(bg_backup)
        
    try:
        # Get image resolution
        with Image.open(source_img_path) as img:
            w, h = img.size
            # Ensure width and height are divisible by 2 for H.264
            w = w - (w % 2)
            h = h - (h % 2)
            resolution = (w, h)
            print(f"Adjusted image resolution (divisible by 2): {resolution}")
            
        audio_duration = _probe_audio_duration(audio_path)
        print(f"Audio duration: {audio_duration:.2f} seconds")
        
        # Load engine
        engine = get_provider(EngineId.LIVE_PORTRAIT)
        engine.load()
        
        try:
            pack_dir = Path("captures/temp_pack")
            pack_dir.mkdir(parents=True, exist_ok=True)
            pack_path = pack_dir / "my_static_avatar.tar"
            
            if pack_path.exists():
                print("Identity cache HIT. Loading compiled pack from captures/temp_pack/my_static_avatar.tar...")
                handle = AvatarIdentityHandle(
                    identity_id=IdentityId("my_static_avatar"),
                    pack_path=pack_path,
                    pack_digest="sha256:temp",
                    prepared_at=None
                )
            else:
                # Prepare identity
                print("Identity cache MISS. Compiling identity from source image...")
                assets = engine.prepare_identity(source_img_path)
                
                # Write assets to a tar file or mock a handle
                import tarfile
                with tarfile.open(pack_path, "w") as tar:
                    for name, data in assets.items():
                        import io
                        tarinfo = tarfile.TarInfo(name=name)
                        tarinfo.size = len(data)
                        tar.addfile(tarinfo, io.BytesIO(data))
                        
                handle = AvatarIdentityHandle(
                    identity_id=IdentityId("my_static_avatar"),
                    pack_path=pack_path,
                    pack_digest="sha256:temp",
                    prepared_at=None
                )
            
            job_id = "static-anim"
            chunk_request = RenderChunkRequest(
                job_id=job_id,
                audio_window=(0.0, audio_duration),
                audio_path=audio_path,
                fps=25,
                resolution=resolution,
                chunk_index=0,
                overlap_seconds=0.0,
                face_region_only=False, # We want full-frame pasting
                face_motion_timeline_path=None
            )
            
            print("Rendering video (1st run - Cold start/Warm-up)...")
            engine.render_chunk(chunk_request, handle)
            
            print("Rendering video (2nd run - Warm GPU)...")
            t0 = time.perf_counter()
            chunk_result = engine.render_chunk(chunk_request, handle)
            wall_seconds = time.perf_counter() - t0
            
            print(f"Warm render completed in {wall_seconds:.2f}s. Muxing audio...")
            mux_audio(chunk_result.output_path, audio_path, output_path)
            
            gpu_seconds = chunk_result.gpu_seconds
            runpod_cost = gpu_seconds * (0.22 / 3600.0)
            aws_cost = gpu_seconds * (1.00 / 3600.0)
            
            print("\n--- Production Summary ---")
            print(f"Output Path: {output_path}")
            print(f"Video Duration: {audio_duration:.2f} seconds")
            print(f"GPU Time: {gpu_seconds:.2f} seconds")
            print("\n--- Estimated Cost ---")
            print(f"RunPod (RTX 4060 Ti @ $0.22/hr): ${runpod_cost:.6f} USD")
            print(f"AWS (A10G @ $1.00/hr):          ${aws_cost:.6f} USD")
            
        finally:
            engine.unload()
            
    finally:
        if bg_backup.exists():
            print("Restoring tv_studio_background.png...")
            bg_backup.rename(bg_path)

if __name__ == "__main__":
    main()
