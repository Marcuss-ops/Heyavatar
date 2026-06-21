from pathlib import Path
from providers.compositing.ffmpeg.compositor import FFmpegPoissonCompositor
from contracts.compositor import CompositingRequest

class CompositionWorker:
    def __init__(self):
        self.compositor = FFmpegPoissonCompositor()

    def process_composite(
        self,
        body_video: str,
        lipsynced_face: str,
        face_mask: str,
        neck_mask: str,
        face_transforms: str,
        color_profile: str
    ) -> dict:
        req = CompositingRequest(
            body_video_path=Path(body_video),
            lipsynced_face_video_path=Path(lipsynced_face),
            face_mask_video_path=Path(face_mask),
            neck_mask_video_path=Path(neck_mask),
            face_transforms_path=Path(face_transforms),
            color_profile_path=Path(color_profile)
        )
        res = self.compositor.composite(req)
        return {
            "status": "composited",
            "composited_video": str(res.composited_video_path)
        }
