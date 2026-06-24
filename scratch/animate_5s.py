import sys
import os
import time
import wave
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
    # 1. Generate 5-second speech audio by concatenating speech.wav 5 times
    speech_source = Path("bench_run/speech.wav")
    audio_path = Path("bench_run/speech_5s.wav")
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    
    with wave.open(str(speech_source), "rb") as w_in:
        params = w_in.getparams()
        frames = w_in.readframes(w_in.getnframes())
        
    with wave.open(str(audio_path), "wb") as w_out:
        w_out.setparams(params)
        for _ in range(5):
            w_out.writeframes(frames)
            
    print("Generated 5-second speech audio by concatenating speech.wav.")

    source_img_path = Path("assets/ChatGPT Image 27 mag 2026, 14_29_11 (2).png")
    output_path = Path("captures/static_animated_5s.mp4")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Rename TV studio background to avoid chroma keying
    bg_path = Path("assets/tv_studio_background.png")
    bg_backup = Path("assets/tv_studio_background_backup.png")
    if bg_path.exists():
        print("Backing up tv_studio_background.png...")
        bg_path.rename(bg_backup)
        
    try:
        with Image.open(source_img_path) as img:
            w_dim, h_dim = img.size
            w_dim = w_dim - (w_dim % 2)
            h_dim = h_dim - (h_dim % 2)
            resolution = (w_dim, h_dim)
            print(f"Adjusted image resolution: {resolution}")
            
        audio_duration = _probe_audio_duration(audio_path)
        print(f"Audio duration: {audio_duration:.2f} seconds")
        
        # Measure Engine Loading Time
        print("Loading engine (LivePortrait)...")
        t_load_start = time.perf_counter()
        engine = get_provider(EngineId.LIVE_PORTRAIT)
        engine.load()
        t_load = time.perf_counter() - t_load_start
        print(f"Engine loaded in {t_load:.2f}s.")
        
        try:
            # Measure Identity Preparation Time
            t_id_start = time.perf_counter()
            pack_dir = Path("captures/temp_pack")
            pack_dir.mkdir(parents=True, exist_ok=True)
            pack_path = pack_dir / "my_new_avatar.tar"
            
            if pack_path.exists():
                print("Identity cache HIT.")
                handle = AvatarIdentityHandle(
                    identity_id=IdentityId("my_new_avatar"),
                    pack_path=pack_path,
                    pack_digest="sha256:new_temp",
                    prepared_at=None
                )
            else:
                print("Identity cache MISS. Compiling identity...")
                assets = engine.prepare_identity(source_img_path)
                import tarfile
                with tarfile.open(pack_path, "w") as tar:
                    for name, data in assets.items():
                        import io
                        tarinfo = tarfile.TarInfo(name=name)
                        tarinfo.size = len(data)
                        tar.addfile(tarinfo, io.BytesIO(data))
                handle = AvatarIdentityHandle(
                    identity_id=IdentityId("my_new_avatar"),
                    pack_path=pack_path,
                    pack_digest="sha256:new_temp",
                    prepared_at=None
                )
            t_id = time.perf_counter() - t_id_start
            print(f"Identity resolved/prepared in {t_id:.2f}s.")
            
            job_id = "static-anim-5s"
            chunk_request = RenderChunkRequest(
                job_id=job_id,
                audio_window=(0.0, audio_duration),
                audio_path=audio_path,
                fps=25,
                resolution=resolution,
                chunk_index=0,
                overlap_seconds=0.0,
                face_region_only=False,
                face_motion_timeline_path=None
            )
            
            print("Rendering video (Cold start / CUDA warmup)...")
            t_cold_start = time.perf_counter()
            engine.render_chunk(chunk_request, handle)
            t_cold = time.perf_counter() - t_cold_start
            print(f"Cold render completed in {t_cold:.2f}s.")
            
            print("Rendering video (Warm GPU / Active rendering)...")
            t_warm_start = time.perf_counter()
            chunk_result = engine.render_chunk(chunk_request, handle)
            t_warm = time.perf_counter() - t_warm_start
            print(f"Warm render completed in {t_warm:.2f}s.")
            
            # Measure Audio Muxing Time
            print("Muxing audio...")
            t_mux_start = time.perf_counter()
            mux_audio(chunk_result.output_path, audio_path, output_path)
            t_mux = time.perf_counter() - t_mux_start
            print(f"Audio muxed in {t_mux:.2f}s.")
            
            gpu_seconds = chunk_result.gpu_seconds
            runpod_cost = gpu_seconds * (0.22 / 3600.0)
            
            print("\n--- Detailed Performance Metrics ---")
            print(f"1. Engine Load Time:           {t_load:.4f} seconds")
            print(f"2. Identity Resolution/Pack:   {t_id:.4f} seconds")
            print(f"3. Cold Render (CUDA Warmup):  {t_cold:.4f} seconds")
            print(f"4. Warm Render (GPU Active):   {t_warm:.4f} seconds")
            print(f"5. Audio Muxing Time:          {t_mux:.4f} seconds")
            print(f"------------------------------------")
            print(f"Total Wall-Time (warm phases): {t_warm + t_mux:.4f} seconds")
            print(f"Active GPU Time (warm render): {gpu_seconds:.4f} seconds")
            print(f"Estimated Cost (RunPod):       ${runpod_cost:.6f} USD")
            
        finally:
            engine.unload()
            
    finally:
        if bg_backup.exists():
            print("Restoring tv_studio_background.png...")
            bg_backup.rename(bg_path)

if __name__ == "__main__":
    main()
