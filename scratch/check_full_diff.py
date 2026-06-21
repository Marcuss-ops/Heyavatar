import numpy as np
from pathlib import Path
from scratch.verify_test import read_frames

def main():
    mp4_path = Path("captures/job-real-gpu-001.mp4")
    if not mp4_path.exists():
        print(f"File not found: {mp4_path}")
        return
    frames = read_frames(mp4_path)
    print(f"Total frames: {len(frames)}")
    if not frames:
        return
    
    # Calculate SSD over the entire frame (all y, all x)
    for i in range(len(frames) - 1):
        diff = frames[i].astype(float) - frames[i+1].astype(float)
        ssd_full = float(np.mean(diff ** 2))
        max_diff = float(np.max(np.abs(diff)))
        print(f"Frame {i} -> {i+1} | Full SSD: {ssd_full:.6f} | Max Diff: {max_diff:.1f}")

if __name__ == "__main__":
    main()
