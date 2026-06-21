"""Checkpoint manifest pins and lightweight ``CheckpointEntry`` dataclass.

The manifest is the single source of truth for the expected set of
weight files, their upstream URLs, and their SHA256 pins. ``CheckpointEntry``
is the runtime-friendly view of one manifest row with a ``local_path``
slot that the manager fills in after a download.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


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
        "sha256": "5279bb8654293dbdf327030b397f107237dd9212fb11dd75b83dfb635211ceb5",
        "size_bytes": 3387959,
    },
    {
        "name": "motion_extractor.pth",
        "url": "https://huggingface.co/KlingTeam/LivePortrait/resolve/main/liveportrait/base_models/motion_extractor.pth",
        "sha256": "251e6a94ad667a1d0c69526d292677165110ef7f0cf0f6d199f0e414e8aa0ca5",
        "size_bytes": 112545506,
    },
    {
        "name": "warping_module.pth",
        "url": "https://huggingface.co/KlingTeam/LivePortrait/resolve/main/liveportrait/base_models/warping_module.pth",
        "sha256": "2f61a6f265fe344f14132364859a78bdbbc2068577170693da57fb96d636e282",
        "size_bytes": 182180086,
    },
    {
        "name": "stitching_retargeting_module.pth",
        "url": "https://huggingface.co/KlingTeam/LivePortrait/resolve/main/liveportrait/retargeting_models/stitching_retargeting_module.pth",
        "sha256": "3652d5a3f95099141a56986aaddec92fadf0a73c87a20fac9a2c07c32b28b611",
        "size_bytes": 2393098,
    },
    {
        "name": "spade_generator.pth",
        "url": "https://huggingface.co/KlingTeam/LivePortrait/resolve/main/liveportrait/base_models/spade_generator.pth",
        "sha256": "4780afc7909a9f84e24c01d73b31a555ef651521a1fe3b2429bd04534d992aee",
        "size_bytes": 221813590,
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
