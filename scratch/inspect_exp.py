import sys
import os
import torch
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sadtalker.src.audio2motion.models import Audio2MotionModel

def main():
    model = Audio2MotionModel.load_default(checkpoint_dir="checkpoints")
    
    # Load the wav
    from providers.liveportrait.audio_bridge.dsp import _read_wav_mono_16bit, _linear_resample
    samples, source_sr = _read_wav_mono_16bit(Path("bench_run/speech_edge.wav"))
    resampled = _linear_resample(samples, source_sr, 16000)
    audio_slice = np.asarray(resampled, dtype=np.float32) / 32768.0
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wav_tensor = torch.from_numpy(audio_slice).unsqueeze(0).to(device)
    
    # Run the model
    with torch.no_grad():
        output = model(wav_tensor)
        print("Output shape:", output.shape)
        
        # Let's inspect the first 10 coefficients of exp_pred
        # To get the raw exp_pred, we can run model.audio2exp_model directly
        # Let's rebuild the batch as in forward
        from src.generate_batch import parse_audio_length, crop_pad_audio, generate_blink_seq_randomly
        wav = wav_tensor.squeeze(0).cpu().numpy()
        expected_frames = max(1, int(round((len(wav) / 16000.0) * 25)))
        num_frames = expected_frames
        wav_length = int(num_frames * (16000 / 25))
        wav = crop_pad_audio(wav, wav_length)
        from src.utils.audio import melspectrogram
        orig_mel = melspectrogram(wav).T
        spec = orig_mel.copy()
        
        indiv_mels = []
        syncnet_mel_step_size = 16
        fps = 25
        for i in range(num_frames):
            start_frame_num = i-2
            start_idx = int(80. * (start_frame_num / float(fps)))
            end_idx = start_idx + syncnet_mel_step_size
            seq = list(range(start_idx, end_idx))
            seq = [ min(max(item, 0), orig_mel.shape[0]-1) for item in seq ]
            m = spec[seq, :]
            indiv_mels.append(m.T)
        indiv_mels = np.asarray(indiv_mels)
        indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1).unsqueeze(0).to(device)
        
        ratio = generate_blink_seq_randomly(num_frames)
        ratio = torch.FloatTensor(ratio).unsqueeze(0).to(device)
        
        ref_coeff = torch.zeros((1, num_frames, 70), dtype=torch.float32).to(device)
        class_val = torch.LongTensor([0]).to(device)
        
        batch = {
            'indiv_mels': indiv_mels,
            'ref': ref_coeff,
            'num_frames': num_frames,
            'ratio_gt': ratio,
            'class': class_val
        }
        
        results_dict_exp = model.audio2exp_model.test(batch)
        exp_pred = results_dict_exp['exp_coeff_pred'].squeeze(0).cpu().numpy()
        
        print("\nCoefficient statistics:")
        for idx in range(64):
            coeff_slice = exp_pred[:, idx]
            var = np.var(coeff_slice)
            mean = np.mean(coeff_slice)
            min_val = np.min(coeff_slice)
            max_val = np.max(coeff_slice)
            if var > 0.001:
                print(f"Index {idx:2d}: mean={mean:6.3f}, var={var:6.3f}, range=[{min_val:6.3f}, {max_val:6.3f}]")

if __name__ == "__main__":
    main()
