from pathlib import Path
from providers.compositing.ffmpeg.compositor import FFmpegPoissonCompositor
from contracts.compositor import CompositeRequest

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
        req = CompositeRequest(
            body_video=Path(body_video),
            generated_face_video=Path(lipsynced_face),
            face_mask_video=Path(face_mask),
            neck_mask_video=Path(neck_mask),
            face_transforms=Path(face_transforms),
            output_path=Path(lipsynced_face).parent / "composited_video.mp4"
        )
        res = self.compositor.composite(req)
        return {
            "status": "composited",
            "composited_video": str(res.output_path)
        }
