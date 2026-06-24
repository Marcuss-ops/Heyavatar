import sys
import os
import time
import asyncio
import subprocess
from pathlib import Path
from PIL import Image
import numpy as np
import edge_tts

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.application.render_video.audio_probe import _probe_audio_duration
from src.application.render_video.face_region import mux_audio
from providers import get_provider
from src.domain.enums import EngineId
from src.domain.types import RenderChunkRequest, AvatarIdentityHandle, IdentityId

async def generate_speech(text: str, voice: str, mp3_path: Path, wav_path: Path):
    print(f"Generating TTS using voice '{voice}'...")
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(mp3_path))
    
    print("Transcoding MP3 to 16-bit mono WAV using FFmpeg...")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(mp3_path),
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(wav_path)
    ]
    subprocess.run(cmd, check=True)
    print("TTS WAV file generated successfully.")

def main():
    # Force the neural audio bridge backend to use SadTalker Audio2Motion for phoneme-accurate lip-sync
    os.environ["HEYAVATAR_AUDIO_BRIDGE_BACKEND"] = "neural"

    text = "Hello. ... I have OPTIMIZED... the GPU rendering! ... Now it is... FANTASTIC!"
    voice = "en-US-AvaNeural"
    
    mp3_path = Path("bench_run/speech_edge.mp3")
    wav_path = Path("bench_run/speech_edge.wav")
    output_path = Path("captures/static_animated.mp4")
    
    # 1. Synthesize audio
    asyncio.run(generate_speech(text, voice, mp3_path, wav_path))
    
    source_img_path = Path("assets/ChatGPT Image 27 mag 2026, 14_29_11 (2).png")
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
            
        audio_duration = _probe_audio_duration(wav_path)
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
            
            # 2. Setup rendering chunk request
            job_id = "static-anim-edge"
            chunk_request = RenderChunkRequest(
                job_id=job_id,
                audio_window=(0.0, audio_duration),
                audio_path=wav_path,
                fps=25,
                resolution=resolution,
                chunk_index=0,
                overlap_seconds=0.0,
                face_region_only=False,
                face_motion_timeline_path=None
            )
            
            print("Rendering video...")
            t_warm_start = time.perf_counter()
            chunk_result = engine.render_chunk(chunk_request, handle)
            t_warm = time.perf_counter() - t_warm_start
            print(f"Render completed in {t_warm:.2f}s.")
            
            # Measure Audio Muxing Time
            print("Muxing audio...")
            t_mux_start = time.perf_counter()
            mux_audio(chunk_result.output_path, wav_path, output_path)
            t_mux = time.perf_counter() - t_mux_start
            print(f"Audio muxed in {t_mux:.2f}s.")
            
            gpu_seconds = chunk_result.gpu_seconds
            runpod_cost = gpu_seconds * (0.22 / 3600.0)
            
            print("\n--- Detailed Performance Metrics (Edge TTS) ---")
            print(f"1. Engine Load Time:           {t_load:.4f} seconds")
            print(f"2. Identity Resolution/Pack:   {t_id:.4f} seconds")
            print(f"3. Render:                     {t_warm:.4f} seconds")
            print(f"4. Audio Muxing Time:          {t_mux:.4f} seconds")
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
