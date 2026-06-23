import unittest
import shutil
from pathlib import Path
import numpy as np
import cv2
from providers.compositing.opencv_face.compositor import OpenCVFaceCompositor
from contracts.compositor import CompositeRequest

class TestGreenScreenComposition(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path("tmp_debug_test")
        self.tmp_dir.mkdir(exist_ok=True)
        
        self.width = 160
        self.height = 120
        self.fps = 25
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        
        # 1. Create a dummy green screen body video
        self.body_video_path = self.tmp_dir / "body.mp4"
        writer = cv2.VideoWriter(str(self.body_video_path), self.fourcc, self.fps, (self.width, self.height))
        for _ in range(5):
            # Pure green frame
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            frame[:, :] = (0, 255, 0)  # BGR Green
            writer.write(frame)
        writer.release()
        
        # 2. Create a dummy face video (blue square)
        self.face_video_path = self.tmp_dir / "face.mp4"
        writer = cv2.VideoWriter(str(self.face_video_path), self.fourcc, self.fps, (self.width, self.height))
        for _ in range(5):
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            frame[:, :] = (255, 0, 0)  # BGR Blue
            writer.write(frame)
        writer.release()
        
        # 3. Create dummy masks
        self.face_mask_path = self.tmp_dir / "face_mask.mp4"
        self.neck_mask_path = self.tmp_dir / "neck_mask.mp4"
        for p in [self.face_mask_path, self.neck_mask_path]:
            writer = cv2.VideoWriter(str(p), self.fourcc, self.fps, (self.width, self.height))
            for _ in range(5):
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                cv2.circle(frame, (self.width//2, self.height//2), 20, (255, 255, 255), -1)
                writer.write(frame)
            writer.release()
            
        # 4. Create face transforms NPZ
        self.transforms_path = self.tmp_dir / "face_transforms.npz"
        np.savez(
            self.transforms_path,
            bbox=np.array([[20, 20, 100, 100]] * 5)
        )
        
        # 5. Output path
        self.output_path = self.tmp_dir / "output.mp4"

    def tearDown(self):
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)

    def test_chroma_key_replacement(self):
        # We ensure the TV studio background exists
        bg_dir = Path("assets")
        bg_dir.mkdir(exist_ok=True)
        bg_path = bg_dir / "tv_studio_background.png"
        
        # Create a red background dummy if not present to ensure test is fully self-contained
        had_bg = bg_path.is_file()
        if not had_bg:
            bg_dummy = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            bg_dummy[:, :] = (0, 0, 255)  # BGR Red
            cv2.imwrite(str(bg_path), bg_dummy)
            
        try:
            compositor = OpenCVFaceCompositor()
            req = CompositeRequest(
                body_video=self.body_video_path,
                generated_face_video=self.face_video_path,
                face_mask_video=self.face_mask_path,
                neck_mask_video=self.neck_mask_path,
                face_transforms=self.transforms_path,
                output_path=self.output_path
            )
            res = compositor.composite(req)
            self.assertTrue(self.output_path.is_file())
            
            # Read output and verify the background was replaced (should NOT be pure green)
            cap = cv2.VideoCapture(str(self.output_path))
            ret, frame = cap.read()
            self.assertTrue(ret)
            cap.release()
            
            # Sample a pixel in the corner (should not be green)
            corner_pixel = frame[5, 5]
            self.assertNotEqual(corner_pixel[1], 255)  # G channel should not be 255
            
        finally:
            if not had_bg and bg_path.is_file():
                bg_path.unlink()

if __name__ == "__main__":
    unittest.main()
