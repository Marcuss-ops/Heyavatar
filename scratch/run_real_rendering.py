import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from providers._ffmpeg import write_dummy_mp4
from src.application.run_cached_avatar_from_text import run_cached_avatar_from_text
from providers import get_provider
from src.domain.enums import EngineId

def prepare_templates(base: Path, avatar_id: str):
    pack = base / avatar_id / "body_cache" / "idle_small"
    pack.mkdir(parents=True, exist_ok=True)
    write_dummy_mp4(pack / "body.mp4", duration=2.0, fps=25, colour="0x00FF00", resolution=(1920, 1080))
    write_dummy_mp4(pack / "face_mask.mp4", duration=2.0, fps=25, colour="0xFFFFFF", resolution=(1920, 1080))
    write_dummy_mp4(pack / "neck_mask.mp4", duration=2.0, fps=25, colour="0xFFFFFF", resolution=(1920, 1080))
    
    np.savez_compressed(
        pack / "face_transforms.npz",
        bbox=np.array([[768, 168, 1152, 560]] * 50, dtype=np.float32),
        affine=np.array([np.eye(3, dtype=np.float32)] * 50),
    )
    (pack / "metadata.json").write_text('{"pose_id":"neutral_desk"}', encoding="utf-8")

def main():
    workdir = Path("body_templates")
    avatar_id = "my_avatar"
    prepare_templates(workdir, avatar_id)
    
    # Run the real LivePortrait engine (or mock if HEYAVATAR_MOCK_ENGINE=1)
    engine = get_provider(EngineId.LIVE_PORTRAIT)
    engine.load()
    try:
        result = run_cached_avatar_from_text(
            text="Ciao a tutti! Oggi parliamo di tre differenze molto importanti.",
            audio_path=Path("bench_run/speech.wav"),
            output_path=Path("captures/text_demo.mp4"),
            avatar_id=avatar_id,
            engine=engine,
            language="it",
            source_image=Path(r"C:\Users\pater\Downloads\ChatGPT Image 27 mag 2026, 14_27_14 (4).png"),
            body_templates_dir=workdir,
            capture_root=Path("captures"),
            mode="timeline",
            motion_style="natural",
        )
        print("Success! Output path:", result.output_path)
        
        # Calculate GPU seconds
        gpu_seconds = sum(
            seg.get("gpu_seconds", 0.0)
            for seg in result.metrics.get("segments", [])
        )
        
        # Standard pricing:
        # 1. RTX 4060 Ti (e.g. RunPod) @ $0.22 / hour
        # 2. A10G (e.g. AWS g5.xlarge) @ $1.00 / hour
        runpod_cost = gpu_seconds * (0.22 / 3600.0)
        aws_cost = gpu_seconds * (1.00 / 3600.0)
        
        print("\n--- Production Summary ---")
        print(f"Output Path: {result.output_path}")
        print(f"Video Duration: {result.render_seconds_total:.2f} seconds")
        print(f"GPU Time: {gpu_seconds:.2f} seconds")
        print("\n--- Estimated Cost ---")
        print(f"RunPod (RTX 4060 Ti @ $0.22/hr): ${runpod_cost:.6f} USD")
        print(f"AWS (A10G @ $1.00/hr):          ${aws_cost:.6f} USD")
    finally:
        engine.unload()

if __name__ == "__main__":
    main()
