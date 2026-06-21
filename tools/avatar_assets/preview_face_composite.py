import os
import argparse
import cv2
import numpy as np
from pathlib import Path

def setup_mediapipe_model() -> Path:
    """Resolve face landmarker task path."""
    task_path = Path("checkpoints/face_landmarker.task")
    if not task_path.is_file():
        raise FileNotFoundError("face_landmarker.task not found under checkpoints/.")
    return task_path

def extract_src_landmarks(face_image_path: Path, task_path: Path):
    """Detect landmarks on the source static face image."""
    import mediapipe as mp
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    face_img = cv2.imread(str(face_image_path))
    if face_img is None:
        raise FileNotFoundError(f"Could not read face image: {face_image_path}")
        
    src_h, src_w = face_img.shape[:2]
    
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(task_path)),
        running_mode=VisionRunningMode.IMAGE,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False
    )
    
    with FaceLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        result = landmarker.detect(mp_image)
        if not result.face_landmarks:
            raise RuntimeError(f"No face detected in source face image: {face_image_path}")
        return face_img, result.face_landmarks[0], src_w, src_h

def composite_preview(body_dir: Path, face_image_path: Path, output_path: Path):
    """Composites a static face onto the precomputed body template using affine warp and masks.

    Deterministic preview only (single static image → body template).
    For the production render path that composes a *generated face
    video* onto the body template, see
    ``src.pipeline.compositor.OpenCVFaceCompositor``.
    """
    task_path = setup_mediapipe_model()
    
    # 1. Load source face details
    print("Detecting landmarks on source face image...")
    face_img, src_lms, src_w, src_h = extract_src_landmarks(face_image_path, task_path)
    
    # Anchor points in MediaPipe Face Mesh
    # 133: left eye inner corner, 362: right eye inner corner, 4: nose tip
    src_pts = np.float32([
        [src_lms[133].x * src_w, src_lms[133].y * src_h],
        [src_lms[362].x * src_w, src_lms[362].y * src_h],
        [src_lms[4].x * src_w, src_lms[4].y * src_h]
    ])
    
    # 2. Load precomputed body caches
    transforms_path = body_dir / "face_transforms.npz"
    if not transforms_path.is_file():
        raise FileNotFoundError(f"Transforms NPZ missing: {transforms_path}")
        
    data = np.load(transforms_path)
    target_landmarks = data["landmarks"]
    
    body_cap = cv2.VideoCapture(str(body_dir / "body.mp4"))
    mask_cap = cv2.VideoCapture(str(body_dir / "face_mask.mp4"))
    
    if not body_cap.isOpened() or not mask_cap.isOpened():
        raise RuntimeError("Could not open body.mp4 or face_mask.mp4 in target directory.")
        
    fps = body_cap.get(cv2.CAP_PROP_FPS)
    width = int(body_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(body_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    print(f"Rendering composite preview to {output_path}...")
    
    frame_idx = 0
    while True:
        ret_b, body_frame = body_cap.read()
        ret_m, mask_frame = mask_cap.read()
        
        if not ret_b or not ret_m:
            break
            
        if frame_idx >= len(target_landmarks):
            break
            
        # Get target keypoints for this frame
        lms_t = target_landmarks[frame_idx]
        dst_pts = np.float32([
            [lms_t[133][0] * width, lms_t[133][1] * height],
            [lms_t[362][0] * width, lms_t[362][1] * height],
            [lms_t[4][0] * width, lms_t[4][1] * height]
        ])
        
        # 3. Compute Affine transform
        M = cv2.getAffineTransform(src_pts, dst_pts)
        warped_face = cv2.warpAffine(face_img, M, (width, height), flags=cv2.INTER_LINEAR)
        
        # 4. Prepare feathered mask (blur the edges for smooth transition)
        mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to create a soft edge
        mask_blurred = cv2.GaussianBlur(mask_gray, (31, 31), 0)
        mask_float = mask_blurred.astype(np.float32) / 255.0
        mask_float = mask_float[..., None]
        
        # 5. Alpha blend
        composite = (mask_float * warped_face + (1.0 - mask_float) * body_frame).astype(np.uint8)
        
        writer.write(composite)
        frame_idx += 1
        
    body_cap.release()
    mask_cap.release()
    writer.release()
    print("Composite preview rendered successfully!")

def main():
    parser = argparse.ArgumentParser(description="Render a static face preview composite.")
    parser.add_argument("--body-dir", required=True, help="Directory containing body.mp4, face_mask.mp4, face_transforms.npz")
    parser.add_argument("--face", required=True, help="Path to static source face PNG/JPG image.")
    parser.add_argument("--output", required=True, help="Path to output video file.")
    args = parser.parse_args()
    
    composite_preview(Path(args.body_dir), Path(args.face), Path(args.output))

if __name__ == "__main__":
    main()
