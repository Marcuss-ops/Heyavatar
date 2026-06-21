from pathlib import Path
from typing import Tuple
from contracts.body_asset_provider import BodyAssetProvider, BodyAsset

class PrerecordedTemplateProvider(BodyAssetProvider):
    def __init__(self, root_dir: Path = Path("avatar_packs")):
        self.root_dir = root_dir

    def resolve_body_asset(
        self,
        avatar_id: str,
        gesture_id: str,
        outfit_id: str,
        camera_id: str,
        lighting_id: str,
        resolution: Tuple[int, int],
        fps: int,
    ) -> BodyAsset:
        pack_dir = self.root_dir / avatar_id / "body_cache" / gesture_id
        pack_dir.mkdir(parents=True, exist_ok=True)
        
        body_video = pack_dir / "body.mp4"
        face_mask = pack_dir / "face_mask.mp4"
        neck_mask = pack_dir / "neck_mask.mp4"
        face_transforms = pack_dir / "face_transforms.npz"
        metadata = pack_dir / "metadata.json"
        
        if not body_video.is_file():
            body_video.write_bytes(b"PRERECORDED BODY CLIP")
            face_mask.write_bytes(b"PRERECORDED FACE MASK")
            neck_mask.write_bytes(b"PRERECORDED NECK MASK")
            metadata.write_text('{"status": "prerecorded_template"}')
            import numpy as np
            np.savez(
                face_transforms,
                bbox=np.array([[0, 0, 100, 100]]),
                affine=np.eye(3)
            )
            
        return BodyAsset(
            body_video_path=body_video,
            face_mask_video_path=face_mask,
            neck_mask_video_path=neck_mask,
            face_transforms_path=face_transforms,
            metadata_path=metadata
        )
