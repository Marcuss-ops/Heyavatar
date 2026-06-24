"""Real-mode chunk rendering for MuseTalk.

Loads the source latent from the identity pack, extracts per-frame
audio features via Whisper, denoises through the UNet, decodes the
latent back to RGB via the VAE, and writes an mp4.

The methods here are attached to :class:`MuseTalkAdapter` at
import time (see end of this file) so callers can use
``adapter._real_render_chunk(...)`` etc. via the instance.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from providers._ffmpeg import FACE_REGION_RESOLUTION, _read_pack_entry
from providers.musetalk.adapter._upstream import _import_torch
from src.motion.face_bias import load_face_motion_timeline, sample_face_motion_biases
from src.core.logging import get_logger
from src.domain.types import (
    AvatarIdentityHandle,
    RenderChunkRequest,
    RenderChunkResult,
)


def _real_render_chunk_impl(
    self,
    request: RenderChunkRequest,
    identity: AvatarIdentityHandle,
    clipped_end: float,
) -> RenderChunkResult:
    """Real-mode: load latent, extract audio features, UNet denoise,
    VAE decode, and encode frames to mp4."""
    torch = _import_torch()
    if torch is None:
        raise RuntimeError("PyTorch not available for render_chunk")

    start, end = request.audio_window
    fps = request.fps
    duration = max(0.5, clipped_end - start)
    num_frames = int(round(duration * fps))

    # ── 1. Load source latent from identity pack ──────────────
    latent_bytes = _read_pack_entry(
        identity.pack_path, "source_latent.bin"
    )
    source_latent = torch.frombuffer(
        latent_bytes, dtype=torch.float16
    ).reshape(1, 4, 64, 64).to(self._torch_device, dtype=torch.float32)

    # ── 2. Extract audio features via Whisper ─────────────────
    audio_features = self._extract_audio_features(
        request.audio_path,
        start_seconds=start,
        end_seconds=end,
        num_frames=num_frames,
        torch=torch,
    )
    face_motion_timeline = load_face_motion_timeline(request.face_motion_timeline_path)
    face_biases = sample_face_motion_biases(
        face_motion_timeline,
        frames=num_frames,
        fps=fps,
    )
    if face_biases["mouth"]:
        scale = torch.as_tensor(
            [1.0 + 0.04 * m + 0.02 * b for m, b in zip(face_biases["mouth"], face_biases["brow"])],
            dtype=torch.float32,
            device=self._torch_device,
        )
        if audio_features.ndim >= 2 and audio_features.shape[0] == scale.shape[0]:
            audio_features = audio_features * scale[:, None]

    # ── 3. UNet denoising ─────────────────────────────────────
    rendered_frames = self._unet_denoise(
        source_latent, audio_features, torch
    )

    # ── 4. VAE decode to RGB ──────────────────────────────────
    rgb_frames = self._vae_decode_frames(rendered_frames, torch)

    # ── 5. Write mp4 ──────────────────────────────────────────
    from providers._ffmpeg import _write_frames_to_mp4
    out_dir = self.settings.capture_dir / request.job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"chunk_{request.chunk_index:04d}.mp4"
    # Face-region-only: skip pasteback, output face crop at 256×256.
    output_resolution = FACE_REGION_RESOLUTION if request.face_region_only else request.resolution
    _write_frames_to_mp4(
        rgb_frames,
        out_path,
        fps=fps,
        target_resolution=output_resolution,
    )

    return RenderChunkResult(
        chunk_index=request.chunk_index,
        output_path=out_path,
        duration_seconds=duration,
        frames_rendered=len(rgb_frames),
        gpu_seconds=max(0.005, duration * 0.008),
        engine_id=self.engine_id,
    )


def _extract_audio_features_impl(
    self,
    audio_path: Path,
    start_seconds: float,
    end_seconds: float,
    num_frames: int,
    torch,
):
    """Extract per-frame Whisper audio features for UNet guidance.

    Tries upstream pipeline; falls back to random features for testing.
    """
    if self._pipeline is not None:
        try:
            extractor = getattr(self._pipeline, "audio_encoder", None)
            if extractor is not None and hasattr(extractor, "extract"):
                features = extractor.extract(
                    audio_path,
                    start=start_seconds,
                    end=end_seconds,
                )
                if features is not None:
                    return torch.as_tensor(
                        features, dtype=torch.float32, device=self._torch_device
                    )
        except Exception as exc:
            get_logger(__name__).warning("Whisper audio extraction failed: %s", exc)

    if self.settings.mock_engine:
        get_logger(__name__).debug("Using random audio features (no Whisper available)")
        return torch.randn(
            num_frames, 384, device=self._torch_device, dtype=torch.float32
        ) * 0.01

    raise RuntimeError(
        "MuseTalk audio encoder (Whisper) is unavailable in real mode. "
        "Install upstream MuseTalk or set HEYAVATAR_MOCK_ENGINE=1."
    )


def _unet_denoise_impl(
    self,
    source_latent,
    audio_features,
    torch,
) -> list:
    """Run the MuseTalk UNet to denoise and produce output latents.

    Processes frames in batches to saturate GPU Tensor Cores.
    Falls back to source latent + noise if UNet is unavailable.
    """
    if self._pipeline is not None:
        try:
            unet = getattr(self._pipeline, "unet", None)
            if unet is not None and hasattr(unet, "forward"):
                num_frames = audio_features.shape[0]
                batch = self.render_batch_size
                latents: list = []
                current = source_latent.clone() + torch.randn_like(
                    source_latent
                ) * 0.02

                for batch_start in range(0, num_frames, batch):
                    batch_end = min(batch_start + batch, num_frames)
                    batch_af = audio_features[batch_start:batch_end]

                    # Try batched UNet forward; fall back to
                    # per-frame if upstream doesn't support batching.
                    try:
                        batch_latent = current.expand(
                            batch_end - batch_start, -1, -1, -1
                        )
                        out = unet(
                            batch_latent,
                            timestep=1,
                            encoder_hidden_states=batch_af,
                        )
                        if isinstance(out, tuple):
                            out = out[0]
                        for j in range(batch_end - batch_start):
                            latents.append(out[j : j + 1].detach())
                    except Exception as batch_exc:
                        get_logger(__name__).debug(
                            "Batched UNet forward failed (will fall back per-frame): %s",
                            batch_exc,
                        )
                        # Per-frame fallback within batch.
                        for i in range(batch_start, batch_end):
                            af = audio_features[i : i + 1]
                            out = unet(
                                current, timestep=1, encoder_hidden_states=af
                            )
                            if isinstance(out, tuple):
                                out = out[0]
                            latents.append(out.detach())
                return latents
        except Exception as exc:
            get_logger(__name__).warning("UNet denoise failed: %s", exc)

    if self.settings.mock_engine:
        get_logger(__name__).debug("Using random rendered latents (no UNet available)")
        num_frames = audio_features.shape[0]
        return [
            source_latent.clone()
            + torch.randn_like(source_latent) * 0.05
            for _ in range(num_frames)
        ]

    raise RuntimeError(
        "MuseTalk UNet is unavailable in real mode. "
        "Install upstream MuseTalk or set HEYAVATAR_MOCK_ENGINE=1."
    )


def _vae_decode_frames_impl(
    self, latents: list, torch
) -> list:
    """VAE-decode a list of latent tensors to RGB frames.

    Decodes in batches to reduce GPU kernel launch overhead.
    Returns list of ``[H, W, 3]`` numpy uint8 arrays.
    """
    frames: list = []
    vae_available = False
    if self._pipeline is not None:
        try:
            vae = getattr(self._pipeline, "vae", None)
            if vae is not None and hasattr(vae, "decode"):
                vae_available = True
        except Exception:
            pass

    batch = self.render_batch_size
    for batch_start in range(0, len(latents), batch):
        batch_end = min(batch_start + batch, len(latents))
        batch_latents = latents[batch_start:batch_end]

        if vae_available:
            try:
                # Stack into batch: [B, 4, 64, 64]
                stacked = torch.cat(batch_latents, dim=0)
                decoded = self._pipeline.vae.decode(stacked)
                if isinstance(decoded, tuple):
                    decoded = decoded[0]
                # VAE outputs [B, 3, H, W] (CHW); transpose to HWC.
                decoded_np = (
                    decoded.detach().cpu().numpy()
                    .transpose(0, 2, 3, 1)
                )  # [B, H, W, 3]
                for j in range(decoded_np.shape[0]):
                    frame = (decoded_np[j] * 255).clip(0, 255).astype("uint8")
                    frames.append(frame)
                continue
            except Exception:
                vae_available = False  # degrade after batch failure

        # Per-frame fallback within this batch.
        for latent in batch_latents:
            if vae_available:
                try:
                    decoded = self._pipeline.vae.decode(latent)
                    if isinstance(decoded, tuple):
                        decoded = decoded[0]
                    frame = (
                        decoded.detach()
                        .cpu()
                        .numpy()[0]
                        .transpose(1, 2, 0)
                    )
                except Exception:
                    frame = None
                    vae_available = False
            else:
                frame = None

            if frame is None:
                if self.settings.mock_engine:
                    frame = np.random.randint(
                        0, 255, (256, 256, 3), dtype="uint8"
                    )
                else:
                    raise RuntimeError(
                        "MuseTalk VAE decode failed in real mode. "
                        "Check upstream MuseTalk installation."
                    )
            else:
                frame = (frame * 255).clip(0, 255).astype("uint8")
            frames.append(frame)
    return frames


# ── bind to MuseTalkAdapter ───────────────────────────────────────
def _attach_render_methods():
    from providers.musetalk.adapter.engine import MuseTalkAdapter
    MuseTalkAdapter._real_render_chunk = _real_render_chunk_impl
    MuseTalkAdapter._extract_audio_features = _extract_audio_features_impl
    MuseTalkAdapter._unet_denoise = _unet_denoise_impl
    MuseTalkAdapter._vae_decode_frames = _vae_decode_frames_impl


_attach_render_methods()
