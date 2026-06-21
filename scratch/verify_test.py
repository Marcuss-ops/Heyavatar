import numpy as np
from pathlib import Path
import subprocess

def read_frames(mp4_path: Path):
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-i",
        str(mp4_path),
        "-vf",
        "format=rgb24",
        "-f",
        "rawvideo",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    raw = proc.stdout
    if not raw:
        return []
    
    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0",
        str(mp4_path),
    ]
    probe = subprocess.run(probe_cmd, capture_output=True, check=True)
    w_str, h_str = probe.stdout.decode().strip().split(",")
    width, height = int(w_str), int(h_str)
    frame_size = width * height * 3
    frames = []
    for offset in range(0, len(raw), frame_size):
        chunk = raw[offset : offset + frame_size]
        if len(chunk) != frame_size:
            break
        frames.append(np.frombuffer(chunk, dtype=np.uint8).reshape(height, width, 3))
    return frames

def main():
    mp4_path = Path("captures/job-real-gpu-001.mp4")
    if not mp4_path.exists():
        print(f"File not found: {mp4_path}")
        return
    frames = read_frames(mp4_path)
    print(f"Total frames: {len(frames)}")
    if not frames:
        return
    
    print(f"Frame shape: {frames[0].shape}")
    
    def _lower_face_ssd(prev, curr) -> float:
        h = prev.shape[0]
        y0 = (2 * h) // 3
        a = prev[y0:, :, :].astype(np.float32)
        b = curr[y0:, :, :].astype(np.float32)
        max_diff_lower = np.max(np.abs(a - b))
        ssd = float(np.mean((a - b) ** 2))
        return ssd, max_diff_lower
        
    mid = len(frames) // 2
    for i in range(len(frames) - 1):
        ssd, max_diff_lower = _lower_face_ssd(frames[i], frames[i + 1])
        print(f"Frame {i} -> {i+1} SSD: {ssd:.6f} | Lower Max Diff: {max_diff_lower:.1f}")
        
    diff = np.abs(frames[0].astype(float) - frames[20].astype(float))
    max_idx = np.unravel_index(np.argmax(diff), diff.shape)
    print("Max pixel diff (frame 0 vs 20):", np.max(diff), "at index:", max_idx)
    print("Mean pixel diff (frame 0 vs 20):", np.mean(diff))
    lower_diff = np.abs(frames[0][341:, :, :].astype(float) - frames[20][341:, :, :].astype(float))
    print("Max lower diff (frame 0 vs 20):", np.max(lower_diff))

if __name__ == "__main__":
    main()
