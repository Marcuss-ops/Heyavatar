from pathlib import Path
from src.face.resolver import CachedFaceRenderer
from contracts.face_renderer import FaceRenderRequest

class FaceWorker:
    def __init__(self):
        self.renderer = CachedFaceRenderer()

    def process_face_render(self, avatar_id: str, timeline_path: str, fps: int) -> dict:
        req = FaceRenderRequest(
            avatar_id=avatar_id,
            head_motion_timeline_path=Path(timeline_path),
            fps=fps,
            face_region_only=True
        )
        res = self.renderer.render_face(req)
        return {
            "status": "face_rendered",
            "face_track": str(res.face_track_video_path),
            "face_alpha": str(res.face_alpha_video_path)
        }
