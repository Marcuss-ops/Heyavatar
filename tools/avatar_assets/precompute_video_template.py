import os
import sys
import argparse
import json
import urllib.request
import cv2
import numpy as np
from pathlib import Path

def setup_mediapipe_model() -> Path:
    """Download the MediaPipe Face Landmarker task file if missing."""
    task_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    task_path = Path("checkpoints/face_landmarker.task")
    if not task_path.is_file():
        task_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading face_landmarker.task from {task_url}...")
        urllib.request.urlretrieve(task_url, task_path)
    return task_path

def precompute_template(video_path: Path, output_dir: Path):
    """Processes a raw template video to extract face transforms, face mask, and neck mask."""
    task_path = setup_mediapipe_model()
    
    import mediapipe as mp
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup video writers for masks
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    face_mask_writer = cv2.VideoWriter(str(output_dir / "face_mask.mp4"), fourcc, fps, (width, height), False)
    neck_mask_writer = cv2.VideoWriter(str(output_dir / "neck_mask.mp4"), fourcc, fps, (width, height), False)
    
    # Store results for transforms.npz
    matrices = []
    landmarks_list = []
    bboxes = []
    confidences = []
    timestamps = []
    
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(task_path)),
        running_mode=VisionRunningMode.VIDEO,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True
    )
    
    print(f"Processing video {video_path} ({width}x{height}, {fps} FPS, {total_frames} frames)...")
    
    # Lower jaw indices in MediaPipe Face Mesh
    jaw_indices = [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 361]
    
    last_matrix = np.eye(4)
    last_landmarks = np.zeros((478, 3))
    last_bbox = [0, 0, 0, 0]
    
    with FaceLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            timestamp_ms = int((frame_idx / fps) * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            # Default values
            matrix = last_matrix.copy()
            landmarks = last_landmarks.copy()
            bbox = last_bbox.copy()
            confidence = 0.0
            
            face_mask = np.zeros((height, width), dtype=np.uint8)
            neck_mask = np.zeros((height, width), dtype=np.uint8)
            
            if result.face_landmarks:
                confidence = 1.0
                # Extract first face detected
                lms = result.face_landmarks[0]
                pts = np.array([(lm.x * width, lm.y * height) for lm in lms], dtype=np.int32)
                landmarks = np.array([(lm.x, lm.y, lm.z) for lm in lms])
                
                # Bounding box
                x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
                x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
                bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
                
                # Face Transformation Matrix
                if result.facial_transformation_matrixes:
                    matrix = result.facial_transformation_matrixes[0]
                    
                # 1. Face Mask (convex hull of face landmarks)
                hull = cv2.convexHull(pts)
                cv2.fillConvexPoly(face_mask, hull, 255)
                
                # 2. Neck Mask (jaw points projected downwards)
                jaw_pts = pts[jaw_indices]
                neck_pts = list(jaw_pts)
                neck_pts.append([width - 1, height - 1])
                neck_pts.append([0, height - 1])
                neck_poly = np.array(neck_pts, dtype=np.int32)
                cv2.fillConvexPoly(neck_mask, cv2.convexHull(neck_poly), 255)
                neck_mask = cv2.GaussianBlur(neck_mask, (51, 51), 0)
                
                # Update last known face tracking states
                last_matrix = matrix
                last_landmarks = landmarks
                last_bbox = bbox
                
            face_mask_writer.write(face_mask)
            neck_mask_writer.write(neck_mask)
            
            matrices.append(matrix)
            landmarks_list.append(landmarks)
            bboxes.append(bbox)
            confidences.append(confidence)
            timestamps.append(timestamp_ms)
            
            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"Processed frame {frame_idx}/{total_frames}...")
                
    cap.release()
    face_mask_writer.release()
    neck_mask_writer.release()
    
    # Save transforms npz
    np.savez_compressed(
        output_dir / "face_transforms.npz",
        matrices=np.array(matrices),
        landmarks=np.array(landmarks_list),
        bbox=np.array(bboxes),
        confidence=np.array(confidences),
        timestamp_ms=np.array(timestamps)
    )
    
    # Copy original video as body.mp4
    import shutil
    shutil.copy(video_path, output_dir / "body.mp4")
    
    # Save metadata.json
    metadata = {
        "avatar_id": output_dir.parent.parent.name,
        "gesture_id": output_dir.name,
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": frame_idx,
        "status": "precomputed"
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Precomputation complete! Outputs saved in: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Precompute video template with face transforms and masks.")
    parser.add_argument("--input", required=True, help="Path to raw input template MP4 video.")
    parser.add_argument("--output-dir", required=True, help="Path to output directory inside body cache.")
    args = parser.parse_args()
    
    precompute_template(Path(args.input), Path(args.output_dir))

if __name__ == "__main__":
    main()
