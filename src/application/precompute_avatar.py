from pathlib import Path
from typing import List
from src.body.resolver import CachedBodyAssetProvider

class AvatarPrecomputer:
    def __init__(self, body_provider: CachedBodyAssetProvider):
        self.body_provider = body_provider

    def precompute(self, avatar_id: str, gestures: List[str], render_profile: str) -> dict:
        """Precomputes and caches body templates for onboarding an avatar."""
        results = {}
        for g in gestures:
            asset = self.body_provider.resolve_body_asset(
                avatar_id=avatar_id,
                gesture_id=g,
                outfit_id="business_01",
                camera_id="medium_shot",
                lighting_id="studio",
                resolution=(1920, 1080),
                fps=25
            )
            results[g] = {
                "body_video": str(asset.body_video_path),
                "face_mask": str(asset.face_mask_video_path),
                "status": "cached"
            }
        return results
