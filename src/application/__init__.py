"""Application services — use cases that orchestrate the engine.

These classes are entry points for the workers and the API. They keep
business orchestration out of the routes and out of the workers so each
side stays independent and testable.
"""

from src.application.compile_avatar import AvatarCompiler
from src.application.render_video.config import ChunkConfig
from src.application.render_video.use_case import RenderVideo
from src.application.telemetry import TelemetryRecorder

__all__ = ["AvatarCompiler", "ChunkConfig", "RenderVideo", "TelemetryRecorder"]

# Canonical entrypoint: make this avatar talk (June 2026 consolidation).
# Replaces inlined orchestration in tools/speak_avatar.py and the
# scratch/ scripts (animate_edge_tts.py, run_real_rendering.py, render_user_avatar.py).
from src.application.speak_avatar import (
    SpeakAvatarResult,
    speak_avatar,
    speak_avatar_sync,
)
