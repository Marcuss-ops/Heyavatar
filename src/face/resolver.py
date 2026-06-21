from pathlib import Path
from contracts.face_renderer import FaceRenderer, FaceRenderRequest, FaceRenderResult

class CachedFaceRenderer(FaceRenderer):
    def __init__(self, cache_root: Path = Path("avatar_packs")):
        self.cache_root = cache_root

    def render_face(self, request: FaceRenderRequest) -> FaceRenderResult:
        target_dir = self.cache_root / request.avatar_id / "face_cache"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        face_track = target_dir / "face_track.mp4"
        face_alpha = target_dir / "face_alpha.mp4"
        
        # Mock file writing if they do not exist
        if not face_track.is_file():
            face_track.write_bytes(b"MOCK FACE TRACK")
            face_alpha.write_bytes(b"MOCK FACE ALPHA")
            
        return FaceRenderResult(
            face_track_video_path=face_track,
            face_alpha_video_path=face_alpha
        )
