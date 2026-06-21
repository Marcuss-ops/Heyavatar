import unittest
from pathlib import Path
from src.motion.registry import GestureRegistry
from src.body.registry import BodyProfileRegistry
from workers.planner_worker import PlannerWorker
from workers.avatar_precompute_worker import AvatarPrecomputeWorker
from workers.face_worker import FaceWorker
from workers.lipsync_worker import LipSyncWorker
from workers.composition_worker import CompositionWorker
from workers.quality_worker import QualityWorker

class TestNewArchitecture(unittest.TestCase):
    def test_registries_load(self):
        gesture_reg = GestureRegistry()
        body_reg = BodyProfileRegistry()
        
        # Verify registries parse configuration correctly
        self.assertGreater(len(gesture_reg.list_gestures()), 0)
        self.assertIn("count_three", gesture_reg.gestures)
        self.assertIn("male_business_01", body_reg.bodies)
        self.assertIn("studio_1080p", body_reg.renders)

    def test_pipeline_flow(self):
        # 1. Planner Worker
        planner = PlannerWorker()
        text = "Oggi vedremo tre elementi fondamentali."
        words_timestamps = [{"word": "tre", "start": 1.2, "end": 1.5}]
        plan_res = planner.process_job("avatar_001", text, "it-IT", "voice_12", words_timestamps)
        self.assertEqual(plan_res["status"], "planned")
        self.assertGreater(len(plan_res["timeline"]["segments"]), 0)

        # 2. Avatar Precompute Worker
        precomputer = AvatarPrecomputeWorker()
        pre_res = precomputer.process_precompute("avatar_001", ["idle_small", "count_three"], "studio_1080p")
        self.assertEqual(pre_res["status"], "precomputed")
        self.assertIn("count_three", pre_res["results"])

        # 3. Face Worker
        face_worker = FaceWorker()
        face_res = face_worker.process_face_render("avatar_001", "avatar_packs/avatar_001/body_cache/count_three/face_transforms.npz", 25)
        self.assertEqual(face_res["status"], "face_rendered")

        # 4. LipSync Worker
        lips_worker = LipSyncWorker()
        lips_res = lips_worker.process_lipsync(face_res["face_track"], "captures/modulated_speech.wav")
        self.assertEqual(lips_res["status"], "lipsynced")

        # 5. Composition Worker
        comp_worker = CompositionWorker()
        comp_res = comp_worker.process_composite(
            body_video=pre_res["results"]["count_three"]["body_video"],
            lipsynced_face=lips_res["lipsynced_face"],
            face_mask=pre_res["results"]["count_three"]["face_mask"],
            neck_mask=pre_res["results"]["count_three"]["body_video"], # stub
            face_transforms="avatar_packs/avatar_001/body_cache/count_three/face_transforms.npz",
            color_profile="avatar_packs/avatar_001/identity/color_profile.json"
        )
        self.assertEqual(comp_res["status"], "composited")

        # 6. Quality Worker
        qc_worker = QualityWorker()
        qc_res = qc_worker.process_qc(comp_res["composited_video"], "captures/modulated_speech.wav")
        self.assertEqual(qc_res["status"], "qc_done")
        self.assertTrue(qc_res["passed"])

if __name__ == "__main__":
    unittest.main()
