"""MuseTalk adapter — implements :class:`AvatarEngine` for MuseTalk.

MuseTalk (MIT, https://github.com/TMElyralab/MuseTalk) is a real-time
lip-sync model that operates in the VAE latent space. The pipeline:

1. **Face detection** — MediaPipe or InsightFace detects and aligns the
   face region to a 256×256 canonical crop.
2. **VAE encode** — ``sd-vae-ft-mse`` encodes the face crop into a
   4-channel latent tensor.
3. **Whisper audio** — Whisper (tiny) extracts per-frame audio features
   from the driving audio chunk.
4. **UNet denoise** — The MuseTalk UNet takes the source latent + audio
   features and denoises into a new latent.
5. **VAE decode** — Decode the output latent back to RGB frames.
6. **Paste-back** — Composite the rendered face back onto the original
   frame using the alignment inverse transform.

Mode switch
-----------
``HEYAVATAR_MOCK_ENGINE=1`` (the default in CI) short-circuits to
deterministic synthetic data so the pipeline stays testable without
GPU/weights. When unset, the adapter attempts real imports and
falls back to DEGRADED on failure.

Submodules
----------
* :mod:`checkpoints` — MuseTalk checkpoint manifest and the
  ``MuseTalkCheckpointManager`` subclass.
* :mod:`_upstream` — lazy import helpers (torch, upstream MuseTalk).
* :mod:`_mock` — deterministic mock-mode helpers.
* :mod:`_identity` — real-mode identity prep (face detect + VAE encode).
* :mod:`_render` — real-mode chunk rendering (Whisper + UNet + VAE decode).
* :mod:`engine` — the ``MuseTalkAdapter`` dataclass and its lifecycle.
"""
