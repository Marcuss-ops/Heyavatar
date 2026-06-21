from providers.body_assets.prerecorded.template_provider import PrerecordedTemplateProvider
from src.application.precompute_avatar import AvatarPrecomputer

class AvatarPrecomputeWorker:
    def __init__(self):
        self.provider = PrerecordedTemplateProvider()
        self.precomputer = AvatarPrecomputer(self.provider)

    def process_precompute(self, avatar_id: str, gestures: list, render_profile: str) -> dict:
        results = self.precomputer.precompute(avatar_id, gestures, render_profile)
        return {
            "status": "precomputed",
            "results": results
        }
