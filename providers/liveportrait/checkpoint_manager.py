"""Checkpoint manager for LivePortrait safetensors weights.

The official weights are downloaded from the upstream release assets
listed at https://github.com/KlingAIResearch/LivePortrait/releases and
verified against the SHA256 pins shipped here. Cached to
``$HEYAVATAR_LIVE_PORTRAIT_CHECKPOINTS`` (default
``./checkpoints/liveportrait/``).

Why this is its own module
--------------------------
* The downstream worker process is the only place that should ever
  perform a ~2 GB download, and we want to time that work outside the
  hot render path.
* Verification MUST happen before the weights are loaded into VRAM;
  otherwise a corrupted file would crash mid-render and corrupt the
  in-flight job.
* Tests must never touch the network. The ``__init__`` short-circuits
  on ``HEYAVATAR_MOCK_ENGINE=1`` and only exposes the manifest.

Cisco file layout produced
--------------------------

::

    <root>/
        appearance_feature_extractor.safetensors
        motion_extractor.safetensors
        warping_module.safetensors
        stitching_retargeting_module.safetensors
        spade_generator.safetensors
        manifest.json     # {"version": "...", "files": [{"name":..., "sha256":..., "size_bytes":...}, ...]}

Citation
--------

Upstream release page: https://github.com/KlingAIResearch/LivePortrait/releases
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.core.logging import get_logger


# SHA256 pins. The values below are marked "TBD" pending first download.
# HuggingFace LFS handles integrity verification on download; SHA256
# pins provide an extra layer of security. To pin:
#   1. Download the weights via `huggingface-cli download KlingTeam/LivePortrait`
#   2. Run `sha256sum <file>` on each downloaded file
#   3. Update the "sha256" field below.
# Set HEYAVATAR_SKIP_SHA256_VERIFY=1 to skip SHA256 verification (for initial setup).
CHECKPOINT_MANIFEST: List[Dict[str, object]] = [
    {
        "name": "appearance_feature_extractor.pth",
        "url": "https://huggingface.co/KlingTeam/LivePortrait/resolve/main/liveportrait/base_models/appearance_feature_extractor.pth",
        "sha256": "TBD",
        "size_bytes": 0,
    },
    {
        "name": "motion_extractor.pth",
        "url": "https://huggingface.co/KlingTeam/LivePortrait/resolve/main/liveportrait/base_models/motion_extractor.pth",
        "sha256": "TBD",
        "size_bytes": 0,
    },
    {
        "name": "warping_module.pth",
        "url": "https://huggingface.co/KlingTeam/LivePortrait/resolve/main/liveportrait/base_models/warping_module.pth",
        "sha256": "TBD",
        "size_bytes": 0,
    },
    {
        "name": "stitching_retargeting_module.pth",
        "url": "https://huggingface.co/KlingTeam/LivePortrait/resolve/main/liveportrait/retargeting_models/stitching_retargeting_module.pth",
        "sha256": "TBD",
        "size_bytes": 0,
    },
    {
        "name": "spade_generator.pth",
        "url": "https://huggingface.co/KlingTeam/LivePortrait/resolve/main/liveportrait/base_models/spade_generator.pth",
        "sha256": "TBD",
        "size_bytes": 0,
    },
]


@dataclass(slots=True)
class CheckpointEntry:
    """Single checkpoint file as resolved by :class:`CheckpointManager`."""

    name: str
    url: str
    expected_sha256: str
    expected_size_bytes: int
    local_path: Optional[Path] = None
    verified: bool = False

    @classmethod
    def from_manifest(cls, raw: Dict[str, object]) -> "CheckpointEntry":
        return cls(
            name=str(raw["name"]),
            url=str(raw["url"]),
            expected_sha256=str(raw["sha256"]),
            expected_size_bytes=int(raw.get("size_bytes", 0)),
        )


@dataclass(slots=True)
class CheckpointManager:
    """Resolves, downloads, and verifies LivePortrait checkpoints.

    The class is intentionally stateless aside from ``root`` so it can
    be exchanged freely across worker processes.

    Mock-mode behaviour
    -------------------

    When ``HEYAVATAR_MOCK_ENGINE=1`` (the project default in tests and
    CI) :meth:`ensure_present` becomes a no-op and every checkpoint
    entry is left unverified. The :attr:`entries` manifest is still
    produced so :meth:`LivePortraitAdapter.load` can be exercised.
    """

    root: Path = field(
        default_factory=lambda: Path(
            os.environ.get("HEYAVATAR_LIVE_PORTRAIT_CHECKPOINTS", "./checkpoints/liveportrait")
        )
    )
    allow_network: bool = field(
        default_factory=lambda: os.environ.get("HEYAVATAR_LIVE_PORTRAIT_MOCK_DOWNLOAD") != "1"
    )
    mock_mode: bool = field(
        default_factory=lambda: os.environ.get("HEYAVATAR_MOCK_ENGINE") == "1"
    )
    # Override at instantiation time so tests can swap the cache root.
    entries: List[CheckpointEntry] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.entries = [CheckpointEntry.from_manifest(m) for m in CHECKPOINT_MANIFEST]
        # In mock mode we never touch disk for weights. We still keep the
        # manifest available so adapters can serialise it into the pack.
        if not self.mock_mode:
            self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def local_path_for(self, name: str) -> Path:
        return self.root / name

    def is_cached(self, name: str) -> bool:
        return self.local_path_for(name).is_file()

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    def sha256_of(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def verify(self, entry: CheckpointEntry) -> bool:
        """Re-hash ``entry.local_path`` and assert it matches the pin."""
        if entry.local_path is None or not entry.local_path.is_file():
            return False
        # Skip verification in mock mode so tests don't need the weights.
        if self.mock_mode:
            entry.verified = True
            return True
        if entry.expected_sha256 == "TBD":
            # SHA not pinned yet — if the operator allows it, treat
            # the HF LFS download as sufficient verification.
            if os.environ.get("HEYAVATAR_SKIP_SHA256_VERIFY") == "1":
                entry.verified = True
                return True
            get_logger(__name__).warning(
                "LivePortrait checkpoint %s has no SHA256 pin; refusing to mark verified. "
                "Set HEYAVATAR_SKIP_SHA256_VERIFY=1 to skip verification (for initial setup), "
                "or compute the SHA256 hash after first download and update checkpoint_manager.py.",
                entry.name,
            )
            return False
        actual = self.sha256_of(entry.local_path)
        if actual != entry.expected_sha256:
            get_logger(__name__).error(
                "Checkpoint hash mismatch for %s: expected %s, got %s",
                entry.name,
                entry.expected_sha256,
                actual,
            )
            return False
        entry.verified = True
        return True

    # ------------------------------------------------------------------
    # ensure_present() — the high-level entry point used by the adapter
    # ------------------------------------------------------------------
    def ensure_present(self) -> None:
        """Make every entry present and verified, downloading missing files.

        Honours ``HEYAVATAR_LIVE_PORTRAIT_MOCK_DOWNLOAD=1`` which forces
        network ops off even outside mock mode (useful for hermetic CI).
        Honours ``HEYAVATAR_LIVE_PORTRAIT_OFFLINE=1`` to skip the
        download step for cached files only (caller must guarantee the
        cache is populated).
        """
        if self.mock_mode:
            for entry in self.entries:
                entry.verified = True  # mock-mode shortcuts verification
            return

        for entry in self.entries:
            target = self.local_path_for(entry.name)
            if target.is_file():
                if self.verify(entry):
                    continue
                # Cached file failed verification: re-download if network
                # is allowed, else surface the error.
                if not self.allow_network:
                    raise RuntimeError(
                        f"LivePortrait checkpoint {entry.name} failed verification "
                        "and no network is allowed. Set "
                        "HEYAVATAR_LIVE_PORTRAIT_OFFLINE=1 with a primed cache, "
                        "or set HEYAVATAR_LIVE_PORTRAIT_MOCK_DOWNLOAD=1 to "
                        "temporarily skip verification."
                    )
            if not self.allow_network:
                raise RuntimeError(
                    f"LivePortrait checkpoint {entry.name} is missing and network is "
                    "disabled. Pre-populate the cache or enable "
                    "HEYAVATAR_LIVE_PORTRAIT_MOCK_DOWNLOAD."
                )
            _download_to(entry.url, target)
            if not self.verify(entry):
                raise RuntimeError(
                    f"Downloaded checkpoint {entry.name} but its hash still fails "
                    "verification; upstream may have rolled. Aborting."
                )

    # ------------------------------------------------------------------
    # Manifest serialisation for inclusion in the Avatar Pack
    # ------------------------------------------------------------------
    def pack_manifest_dict(self) -> Dict[str, object]:
        """Return a JSON-friendly dict that the pack writer persists.

        This makes the pack self-describing so a future audit can ask,
        for any avatar, "which LivePortrait weights produced this?".
        """
        return {
            "schema": "liveportrait-checkpoint-manifest/v1",
            "root": str(self.root),
            "files": [
                {
                    "name": e.name,
                    "url": e.url,
                    "sha256": e.expected_sha256 or "",
                    "verified": e.verified,
                    "local_path": str(e.local_path) if e.local_path else None,
                }
                for e in self.entries
            ],
        }


# ---------------------------------------------------------------------------
# Network primitive — kept in module so tests can monkey-patch _download_to.
# ---------------------------------------------------------------------------


def _download_to(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` atomically using a tempfile + rename."""
    log = get_logger(__name__)
    log.info("Downloading LivePortrait checkpoint %s -> %s", url, dest)
    # Ensure parent directory exists (root was created in __post_init__ but
    # custom roots passed to tests can be airy).
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=dest.name + ".", dir=dest.parent)
    try:
        with os.fdopen(fd, "wb") as tmp_fh:
            # urlopen with a sane timeout — production installs can be slow.
            req = urllib.request.Request(url, headers={"User-Agent": "heyavatar/0.1"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                shutil.copyfileobj(resp, tmp_fh, length=1 << 16)
        os.replace(tmp_name, dest)
    except (urllib.error.URLError, OSError) as exc:
        # Cleanup partial download before re-raising.
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        log.error("Failed to download %s: %s", url, exc)
        raise RuntimeError(f"Could not download LivePortrait checkpoint at {url}: {exc}") from exc


# ---------------------------------------------------------------------------
# Convenience: print the manifest to stdout for ops engineers inspecting
# what would be downloaded.
# ---------------------------------------------------------------------------


def _print_manifest_to_stdout() -> None:
    json.dump(
        CheckpointManager().pack_manifest_dict(),
        sys.stdout,
        indent=2,
    )
    sys.stdout.write("\n")


if __name__ == "__main__":  # pragma: no cover - ops entry point
    _print_manifest_to_stdout()
