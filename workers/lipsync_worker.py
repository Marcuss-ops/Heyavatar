from pathlib import Path
from providers.lipsync.musetalk.lip_sync import MuseTalkLipSyncEngine
from contracts.lip_sync_engine import LipSyncRequest

class LipSyncWorker:
    def __init__(self):
        self.engine = MuseTalkLipSyncEngine()

    def process_lipsync(self, face_video_path: str, audio_path: str) -> dict:
        req = LipSyncRequest(
            face_video_path=Path(face_video_path),
            audio_path=Path(audio_path)
        )
        res = self.engine.sync_lips(req)
        return {
            "status": "lipsynced",
            "lipsynced_face": str(res.lipsynced_face_video_path)
        }
