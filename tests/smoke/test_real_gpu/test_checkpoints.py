"""Checkpoint file presence and SHA256 pinning.

Verifies that all 5 LivePortrait checkpoint files are reachable and
non-empty, then prints their actual SHA256 hashes so an operator can
pin them in :data:`CHECKPOINT_MANIFEST`.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from providers.liveportrait.checkpoint_manager.manifest import CHECKPOINT_MANIFEST
from tests.smoke.test_real_gpu._helpers import requires_cuda


@requires_cuda
def test_checkpoints_present_and_readable():
    """Verify all 5 LivePortrait checkpoint files exist and are non-empty."""
    root = Path(
        os.environ.get(
            "HEYAVATAR_LIVE_PORTRAIT_CHECKPOINTS",
            "./checkpoints/liveportrait",
        )
    )
    assert root.is_dir(), f"Checkpoint root missing: {root}"

    total_mb = 0
    for entry in CHECKPOINT_MANIFEST:
        name = entry["name"]
        path = root / name
        assert path.is_file(), f"Missing checkpoint: {name} at {path}"
        size = path.stat().st_size
        assert size > 0, f"Checkpoint {name} is empty"
        total_mb += size
        print(f"  {name}: {size / (1<<20):.1f} MB")
    print(f"  Total: {total_mb / (1<<20):.0f} MB")


@requires_cuda
def test_checkpoint_sha256_verification():
    """Compute SHA256 of each checkpoint file for pinning."""
    root = Path(
        os.environ.get(
            "HEYAVATAR_LIVE_PORTRAIT_CHECKPOINTS",
            "./checkpoints/liveportrait",
        )
    )
    print("\nSHA256 values (copy to checkpoint_manager.py):")
    for entry in CHECKPOINT_MANIFEST:
        name = entry["name"]
        path = root / name
        if not path.is_file():
            print(f"  {name}: MISSING")
            continue
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        print(f'  "{name}": "{h.hexdigest()}"')
    print("Set HEYAVATAR_SKIP_SHA256_VERIFY=1 until these are pinned.")
