"""MuseTalk checkpoint manifest and manager.

The raw manifest pins are conservative placeholders: HuggingFace LFS
handles integrity verification on download, but SHA256 is the
last-line-of-defence hash. Download weights via
``huggingface-cli download TMElyralab/MuseTalk``, run ``sha256sum`` on
each file, and update the ``sha256`` field.
Set ``HEYAVATAR_SKIP_SHA256_VERIFY=1`` to skip verification for
initial setup only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# SHA256 values are "TBD" pending first download.
# HuggingFace LFS handles integrity verification on download.
# To pin: download via `huggingface-cli download TMElyralab/MuseTalk`,
# run `sha256sum` on each file, update the "sha256" field below.
# Set HEYAVATAR_SKIP_SHA256_VERIFY=1 to skip SHA256 verification (for initial setup).
MUSETALK_CHECKPOINT_MANIFEST: list = [
    {
        "name": "musetalk_unet.pth",
        "url": "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/unet.pth",
        "sha256": "TBD",
        "size_bytes": 0,
    },
    {
        "name": "musetalk_vae.bin",
        "url": "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/sd-vae/diffusion_pytorch_model.bin",
        "sha256": "TBD",
        "size_bytes": 0,
    },
    {
        "name": "musetalk_whisper_tiny.bin",
        "url": "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/whisper/pytorch_model.bin",
        "sha256": "TBD",
        "size_bytes": 0,
    },
]


@dataclass(slots=True)
class MuseTalkCheckpointManager:
    """Checkpoint manager tailored for MuseTalk checkpoint paths.

    Subclasses the LivePortrait :class:`CheckpointManager` only at the
    dataclass level — the actual weight resolution happens through the
    inherited ``ensure_present`` / ``local_path_for`` methods.
    """

    root: Path = field(
        default_factory=lambda: Path(
            os.environ.get("HEYAVATAR_MUSETALK_CHECKPOINTS", "./checkpoints/musetalk")
        )
    )
    mock_mode: bool = field(
        default_factory=lambda: os.environ.get("HEYAVATAR_MOCK_ENGINE") == "1"
    )
    entries: list = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Use the parent CheckpointEntry class to build MuseTalk entries.
        from providers.liveportrait.checkpoint_manager.manager import CheckpointManager
        # Need CheckpointEntry; instantiated indirectly via CheckpointManager.
        from providers.liveportrait.checkpoint_manager.manifest import CheckpointEntry
        self.entries = [
            CheckpointEntry.from_manifest(m) for m in MUSETALK_CHECKPOINT_MANIFEST
        ]
        if not self.mock_mode:
            self.root.mkdir(parents=True, exist_ok=True)

    # Re-export the parent's download/verify methods so callers can use
    # ``MuseTalkCheckpointManager`` polymorphically.
    def ensure_present(self) -> None:
        """Download + verify every MuseTalk checkpoint entry.

        The parent ``CheckpointManager`` does the heavy lifting; we
        compose with it so the download bookkeeping and SHA policy
        stay in one place. ``verify()`` is also exposed explicitly so
        callers (and tests) can re-check cached files without
        re-downloading.
        """
        from providers.liveportrait.checkpoint_manager.manager import CheckpointManager
        manager = CheckpointManager(root=self.root, mock_mode=self.mock_mode)
        manager.entries = self.entries
        manager.ensure_present()
        # Mirror verified flags back so callers can introspect.
        for ours, theirs in zip(self.entries, manager.entries):
            ours.verified = theirs.verified

    def verify(self) -> List[bool]:
        """Re-verify already-cached checkpoints against their SHA pins.

        Returns the per-entry verification result without re-downloading.
        Used by the smoke tests and by an operator who wants to audit
        a checkpoint cache without network egress.

        Behaviour is gated on the same env flags as the LivePortrait
        manager:

        * ``HEYAVATAR_MOCK_ENGINE=1`` → everything is reported verified.
        * ``HEYAVATAR_SKIP_SHA256_VERIFY=1`` → SHA-pinned entries whose
          pin is still "TBD" are accepted (dev convenience for first
          setup); if pin is set and the file matches → still ok.
        * Otherwise a ``TBD`` pin blocks verification until the operator
          either pins the actual hash or sets the skip env.
        """
        from src.core.logging import get_logger

        log = get_logger(__name__)
        if self.mock_mode:
            for entry in self.entries:
                entry.verified = True
            return [True] * len(self.entries)

        skip = os.environ.get("HEYAVATAR_SKIP_SHA256_VERIFY") == "1"
        # ONE transient parent manager reused for every entry's hash
        # computation — no per-entry instantiation.
        sha256_of = _shared_sha256_helper()
        results: List[bool] = []
        for entry in self.entries:
            target = self.root / entry.name
            entry.local_path = target
            if not target.is_file():
                log.warning("MuseTalk checkpoint missing: %s", entry.name)
                entry.verified = False
                results.append(False)
                continue
            if entry.expected_sha256 == "TBD":
                if skip:
                    log.info(
                        "MuseTalk checkpoint %s pin is TBD; "
                        "HEYAVATAR_SKIP_SHA256_VERIFY=1 accepted.",
                        entry.name,
                    )
                    entry.verified = True
                else:
                    log.warning(
                        "MuseTalk checkpoint %s has no SHA256 pin. "
                        "Either pin the actual hash or set "
                        "HEYAVATAR_SKIP_SHA256_VERIFY=1 for first-run setup.",
                        entry.name,
                    )
                    entry.verified = False
            else:
                actual = sha256_of(target)
                entry.verified = actual == entry.expected_sha256
                if not entry.verified:
                    log.error(
                        "MuseTalk checkpoint %s hash mismatch: expected %s, got %s",
                        entry.name, entry.expected_sha256, actual,
                    )
            results.append(entry.verified)
        return results


def _shared_sha256_helper():
    """Return the parent's :func:`sha256_of` as a bound callable.

    Avoids reconstructing a :class:`CheckpointManager` for every entry
    verified — we only need its stateless hashing helper, so we keep a
    single throwaway instance around for the lifetime of a ``verify()``
    call.
    """
    from providers.liveportrait.checkpoint_manager.manager import CheckpointManager
    holder = CheckpointManager(root=Path("/dev/null"), mock_mode=False)
    return holder.sha256_of
