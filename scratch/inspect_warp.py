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
    os.environ["HEYAVATAR_MOCK_ENGINE"] = "0"
    get_settings.cache_clear()
    
    engine = get_provider(EngineId.LIVE_PORTRAIT)
    engine.load()
    
    tmp_path = Path("tmp_debug_test")
    tmp_path.mkdir(exist_ok=True)
    
    # Compile identity
    from PIL import Image
    src_jpg = liveportrait_path / "assets" / "examples" / "source" / "s0.jpg"
    actor_path = tmp_path / "actor.png"
    Image.open(src_jpg).save(actor_path)
    
    pack_root = tmp_path / "packs"
    compiler = AvatarCompiler(engine=engine, pack_root=pack_root)
    identity_handle = compiler.compile(IdentitySpec(source_image=actor_path, display_name="Actor 1"))
    
    # Get audio
    audio_path = tmp_path / "speech.wav"
    from tests.smoke.test_real_gpu._helpers import _test_audio
    _test_audio(tmp_path)
    
    driving = audio_to_driving(audio_path, start_seconds=0.0, end_seconds=1.0, fps=25)
    f_s, kp_s, _exp_s = _load_source_bundle(identity_handle.pack_path, torch, engine._torch_device)
    
    kp_d = _build_driving_keypoints(driving, kp_s, torch, engine._torch_device, engine._wrapper)
    kp_d_np = np.asarray(kp_d, dtype=np.float32)
    
    kp_d_batch = torch.as_tensor(kp_d_np, dtype=torch.float32, device=engine._torch_device)
    f_s_batch = f_s.expand(driving.frames, -1, -1, -1, -1)
    kp_s_batch = kp_s.expand(driving.frames, -1, -1)
    
    f_s_b2 = torch.cat([f_s_batch[0:1], f_s_batch[20:21]], dim=0)
    kp_s_b2 = torch.cat([kp_s_batch[0:1], kp_s_batch[20:21]], dim=0)
    kp_d_b2 = torch.cat([kp_d_batch[0:1], kp_d_batch[20:21]], dim=0)
    
    # Now run warping module as a batch of 2
    with torch.no_grad(), engine._wrapper.inference_ctx():
        warp_out_b2 = engine._wrapper.warping_module(f_s_b2, kp_source=kp_s_b2, kp_driving=kp_d_b2)
        
        warp_diff = torch.abs(warp_out_b2['out'][0] - warp_out_b2['out'][1])
        print("B2 Warping output difference (0 vs 20) max:", warp_diff.max().item(), "mean:", warp_diff.mean().item())
        
        # Generator
        gen_b2 = engine._wrapper.spade_generator(feature=warp_out_b2['out'])
        
        gen_diff = torch.abs(gen_b2[0] - gen_b2[1])
        print("B2 Generator output difference (0 vs 20) max:", gen_diff.max().item(), "mean:", gen_diff.mean().item())
        
        # Final output parse
        img_0 = engine._wrapper.parse_output(gen_b2[0:1])[0]
        img_20 = engine._wrapper.parse_output(gen_b2[1:2])[0]
        
        img_diff = np.abs(img_0.astype(float) - img_20.astype(float))
        print("B2 Final parsed image diff (0 vs 20) max:", np.max(img_diff), "mean:", np.mean(img_diff))

if __name__ == "__main__":
    main()
