"""Canonical async entrypoint for the wide speaking-avatar API."""

from __future__ import annotations

import asyncio
import hashlib
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from providers import get_provider
from src.application.avatar_image import load_avatar_source_image
from src.application.render_video.audio_probe import _probe_audio_duration
from src.application.render_video.face_region import mux_audio
from src.application.speech_synthesis import synthesize_speech_to_wav
from src.core.config import get_settings
from src.domain.enums import EngineId
from src.domain.types import AvatarIdentityHandle, IdentityId, RenderChunkRequest

_CANONICAL_TO_ENGINE = {
    "musetalk-v1": EngineId.MUSE_TALK,
    "liveportrait-human-v1": EngineId.LIVE_PORTRAIT,
    "echomimic-v1": EngineId.ECHO_MIMIC,
}

_DEFAULT_SOURCE_IMAGE = Path("assets/green_screen_avatar.jpg")


def _resolve_engine_id(name: str) -> EngineId:
    cleaned = name.strip().lower()
    if cleaned in _CANONICAL_TO_ENGINE:
        return _CANONICAL_TO_ENGINE[cleaned]
    return EngineId(cleaned)


@dataclass(slots=True)
class SpeakAvatarResult:
    """Result of a single speak_avatar() call."""

    text: str
    voice: str
    avatar_id: Optional[str]
    audio_path: Path
    video_path: Optional[Path]
    engine: Optional[EngineId]
    status: str
    mocked: bool = False
    fallback_gesture_used: bool = False
    error: Optional[str] = None
    metrics: dict = field(default_factory=dict)


def _load_or_prepare_identity(
    *,
    engine: object,
    avatar_id: str,
    source_image: Path,
    capture_root: Path,
) -> AvatarIdentityHandle:
    pack_dir = capture_root / "temp_pack"
    pack_dir.mkdir(parents=True, exist_ok=True)
    source_digest = hashlib.sha256(source_image.read_bytes()).hexdigest()[:12]
    pack_path = pack_dir / f"{avatar_id}-{source_digest}.tar"
    if pack_path.exists():
        return AvatarIdentityHandle(
            identity_id=IdentityId(avatar_id),
            pack_path=pack_path,
            pack_digest="sha256:temp",
            prepared_at=None,
        )

    assets = engine.prepare_identity(source_image)
    import io
    import tarfile

    with tarfile.open(pack_path, "w") as tar:
        for name, data in assets.items():
            tarinfo = tarfile.TarInfo(name=name)
            tarinfo.size = len(data)
            tar.addfile(tarinfo, io.BytesIO(data))

    return AvatarIdentityHandle(
        identity_id=IdentityId(avatar_id),
        pack_path=pack_path,
        pack_digest="sha256:temp",
        prepared_at=None,
    )


async def speak_avatar(
    text: str,
    *,
    voice: str = "it-IT-DiegoNeural",
    avatar_id: str = "my_avatar",
    source_image: Optional[str | Path] = _DEFAULT_SOURCE_IMAGE,
    engine: Optional[object] = None,
    language: str = "it",
    fps: int = 25,
    capture_root: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
    audio: Optional[str | Path] = None,
    audio_bridge_backend: str = "dsp",
    manage_engine: bool = True,
) -> SpeakAvatarResult:
    """Canonical async entrypoint: render the wide speaking avatar video."""
    if audio_bridge_backend not in ("dsp", "neural", "auto"):
        raise ValueError("audio_bridge_backend must be dsp|neural|auto")

    os.environ["HEYAVATAR_AUDIO_BRIDGE_BACKEND"] = audio_bridge_backend
    settings = get_settings()
    capture_root_path = Path(capture_root) if capture_root else Path(settings.capture_dir)
    capture_root_path.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path) if output_path else (capture_root_path / ("talk-" + uuid.uuid4().hex[:10] + ".mp4"))
    audio_path = Path(audio) if audio else (capture_root_path / ("talk-" + uuid.uuid4().hex[:10] + ".wav"))

    if audio is None:
        await synthesize_speech_to_wav(text, audio_path, voice=voice)

    engine_instance = engine
    engine_id_resolved: Optional[EngineId] = None
    owns_engine = False
    if engine_instance is None or isinstance(engine_instance, str):
        engine_name = engine_instance if isinstance(engine_instance, str) and engine_instance else "liveportrait-human-v1"
        engine_id_resolved = _resolve_engine_id(engine_name)
        engine_instance = get_provider(engine_id_resolved)
        owns_engine = manage_engine and hasattr(engine_instance, "load")
        if owns_engine:
            engine_instance.load()

    try:
        source_path = Path(source_image) if source_image else _DEFAULT_SOURCE_IMAGE
        if not source_path.is_file():
            raise FileNotFoundError(f"Source image not found: {source_path}")

        if engine_instance is None:
            raise RuntimeError("Engine provider is required")

        audio_duration = _probe_audio_duration(audio_path)
        source_rgb = load_avatar_source_image(source_path)
        resolution = tuple(int(v) for v in source_rgb.size)
        handle = _load_or_prepare_identity(
            engine=engine_instance,
            avatar_id=avatar_id,
            source_image=source_path,
            capture_root=capture_root_path,
        )

        chunk_request = RenderChunkRequest(
            job_id=f"static-{uuid.uuid4().hex[:10]}",
            audio_window=(0.0, audio_duration),
            audio_path=audio_path,
            fps=fps,
            resolution=resolution,
            chunk_index=0,
            overlap_seconds=0.0,
            face_region_only=False,
            face_motion_timeline_path=None,
        )
        chunk_result = engine_instance.render_chunk(chunk_request, handle)
        mux_audio(chunk_result.output_path, audio_path, output_path)
        return SpeakAvatarResult(
            text=text,
            voice=voice,
            avatar_id=avatar_id,
            audio_path=audio_path,
            video_path=output_path,
            engine=engine_id_resolved,
            status="ok",
            mocked=False,
            fallback_gesture_used=False,
            error=None,
            metrics={
                "job_id": chunk_request.job_id,
                "resolution": list(resolution),
                "fps": fps,
                "audio_duration_seconds": round(audio_duration, 3),
            },
        )
    except Exception as exc:
        return SpeakAvatarResult(
            text=text,
            voice=voice,
            avatar_id=avatar_id,
            audio_path=audio_path,
            video_path=None,
            engine=engine_id_resolved,
            status="failed",
            mocked=False,
            fallback_gesture_used=False,
            error="render_failed: " + type(exc).__name__ + ": " + str(exc),
        )
    finally:
        if owns_engine and engine_instance is not None and hasattr(engine_instance, "unload"):
            try:
                engine_instance.unload()
            except Exception:
                pass


def speak_avatar_sync(*args, **kwargs):
    """Synchronous wrapper around speak_avatar (for CLI use)."""
    return asyncio.run(speak_avatar(*args, **kwargs))


__all__ = ["speak_avatar", "speak_avatar_sync", "SpeakAvatarResult"]
