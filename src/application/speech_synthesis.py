"""Speech synthesis helpers for avatar demos.

The primary path uses ``edge_tts`` when available and converts the
generated MP3 to the mono 16 kHz WAV expected by the render pipeline.
On Windows, a PowerShell/SAPI fallback is available so the demo can
still produce voice audio without any extra package.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
from pathlib import Path


async def synthesize_speech_to_wav(
    text: str,
    wav_path: Path,
    *,
    voice: str = "it-IT-DiegoNeural",
) -> Path:
    """Synthesize ``text`` into a mono 16 kHz WAV file."""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    mp3_path = wav_path.with_suffix(".mp3")

    if _has_edge_tts():
        await _edge_tts_to_wav(text=text, voice=voice, mp3_path=mp3_path, wav_path=wav_path)
        return wav_path

    if _has_powershell_speech():
        _powershell_tts_to_wav(text=text, voice=voice, wav_path=wav_path)
        return wav_path

    raise RuntimeError(
        "No TTS backend available. Install edge-tts or run on Windows with PowerShell speech support."
    )


def _has_edge_tts() -> bool:
    try:
        import edge_tts  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


def _has_powershell_speech() -> bool:
    return shutil.which("powershell") is not None or shutil.which("pwsh") is not None


async def _edge_tts_to_wav(*, text: str, voice: str, mp3_path: Path, wav_path: Path) -> None:
    import edge_tts  # type: ignore

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(mp3_path))

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to convert edge_tts output to WAV")

    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(mp3_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True)


def _powershell_tts_to_wav(*, text: str, voice: str, wav_path: Path) -> None:
    shell = shutil.which("powershell") or shutil.which("pwsh")
    if shell is None:
        raise RuntimeError("PowerShell not found")

    script = rf"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = 0
$synth.Volume = 100
try {{
  $synth.SelectVoice('{voice}')
}} catch {{
  # Keep the default voice if the requested one is not installed.
}}
$synth.SetOutputToWaveFile('{wav_path}')
$synth.Speak(@'
{text}
'@)
$synth.Dispose()
"""
    subprocess.run([shell, "-NoProfile", "-Command", script], check=True)


def synthesize_speech(text: str, wav_path: Path, *, voice: str = "it-IT-DiegoNeural") -> Path:
    """Synchronous wrapper around :func:`synthesize_speech_to_wav`."""
    return asyncio.run(synthesize_speech_to_wav(text, wav_path, voice=voice))


__all__ = ["synthesize_speech", "synthesize_speech_to_wav"]
