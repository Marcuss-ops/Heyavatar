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
from src.domain.enums import EngineId
from src.core.config import get_settings
from src.domain.types import IdentitySpec
from src.application.compile_avatar import AvatarCompiler
from providers.liveportrait.audio_bridge.bridge import audio_to_driving
from providers.liveportrait.adapter._render import _load_source_bundle, _build_driving_keypoints

def main():
    # Force real mode
    os.environ["HEYAVATAR_MOCK_ENGINE"] = "0"
    get_settings.cache_clear()
    
    engine = get_provider(EngineId.LIVE_PORTRAIT)
    engine.load()
    
    # Create a temporary path
    tmp_path = Path("tmp_debug_test")
    tmp_path.mkdir(exist_ok=True)
    
    # 1. Compile identity
    from PIL import Image
    src_jpg = liveportrait_path / "assets" / "examples" / "source" / "s0.jpg"
    actor_path = tmp_path / "actor.png"
    Image.open(src_jpg).save(actor_path)
    
    pack_root = tmp_path / "packs"
    compiler = AvatarCompiler(engine=engine, pack_root=pack_root)
    identity_handle = compiler.compile(IdentitySpec(source_image=actor_path, display_name="Actor 1"))
    
    # 2. Get audio
    # Create fake audio
    audio_path = tmp_path / "speech.wav"
    from tests.smoke.test_real_gpu._helpers import _test_audio
    _test_audio(tmp_path) # writes to tmp_path/speech.wav
    
    # 3. Driving
    driving = audio_to_driving(
        audio_path,
        start_seconds=0.0,
        end_seconds=1.0,
        fps=25
    )
    
    print(f"Driving mouth aperture max: {max(driving.mouth_aperture)}")
    print(f"Driving mouth aperture min: {min(driving.mouth_aperture)}")
    
    f_s, kp_s, _exp_s = _load_source_bundle(identity_handle.pack_path, torch, engine._torch_device)
    # Print deltas details
    apertures = torch.tensor(driving.mouth_aperture, dtype=torch.float32, device=engine._torch_device)
    lip_close_ratios = torch.zeros((driving.frames, 2), dtype=torch.float32, device=engine._torch_device)
    lip_close_ratios[:, 0] = 0.15
    lip_close_ratios[:, 1] = 0.15 + apertures * 0.55
    kp_s_expanded = kp_s.expand(driving.frames, -1, -1)
    
    lip_deltas = engine._wrapper.retarget_lip(kp_s_expanded, lip_close_ratios)
    eye_close_ratios = torch.zeros((driving.frames, 3), dtype=torch.float32, device=engine._torch_device)
    eye_deltas = engine._wrapper.retarget_eye(kp_s_expanded, eye_close_ratios)
    
    print(f"lip_deltas min: {lip_deltas.min().item()}, max: {lip_deltas.max().item()}")
    print(f"eye_deltas min: {eye_deltas.min().item()}, max: {eye_deltas.max().item()}")
    
    kp_d = _build_driving_keypoints(
        driving, kp_s, torch, engine._torch_device, engine._wrapper
    )
    
    print(f"kp_d shape: {kp_d.shape}")
    # Compare first frame and last frame of kp_d
    diff = np.abs(kp_d[0] - kp_d[-1])
    print(f"Max absolute diff in driving keypoints between first and last frame: {np.max(diff)}")
    print(f"Mean absolute diff: {np.mean(diff)}")
    
if __name__ == "__main__":
    main()
