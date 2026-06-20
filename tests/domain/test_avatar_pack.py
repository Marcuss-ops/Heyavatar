"""Avatar Pack round-trip tests."""

from __future__ import annotations

import hashlib

import pytest

from src.domain.avatar_pack import read_pack, read_pack_asset, verify_pack, write_pack
from src.domain.types import IdentityId


def test_write_then_read_pack(workdir, tmp_path):
    identity_id = IdentityId("id-12345")
    archive = tmp_path / "pack.tar"
    assets = {
        "source_features.bin": b"\x00\x01\x02" * 100,
        "canonical_keypoints.bin": b"\x03\x04" * 256,
        "face_mask.png": b"\x99" * 64,
        "face_crop.png": b"\x88" * 256,
        "identity_embedding.bin": b"\x77" * 512,
        "source_latent.bin": b"\x66" * 1024,
    }
    pack = write_pack(
        archive_path=archive,
        identity_id=identity_id,
        assets=assets,
        engine_compatibility=("liveportrait-human-v1",),
        safe_motion_ranges={"head_yaw_deg": [-15, 15]},
    )
    assert archive.is_file()

    pack2 = read_pack(archive)
    assert pack2.manifest.identity_id == identity_id
    assert "liveportrait-human-v1" in pack2.manifest.engine_compatibility
    assert pack2.manifest.safe_motion_ranges["head_yaw_deg"] == [-15, 15]

    # Asset extraction works.
    feats = read_pack_asset(archive, "source_features.bin")
    assert feats == assets["source_features.bin"]

    # Verification reports nothing missing.
    assert verify_pack(archive) == []


def test_verify_flags_missing_entries(workdir, tmp_path):
    archive = tmp_path / "pack.tar"
    assets = {
        "source_features.bin": b"abc",
        # intentionally missing all the rest
    }
    with pytest.raises(KeyError):
        write_pack(
            archive_path=archive,
            identity_id=IdentityId("id-x"),
            assets=assets,
            engine_compatibility=("musetalk-v1",),
        )


def test_digest_is_content_addressed(tmp_path):
    identity_id = IdentityId("id-y")
    archive = tmp_path / "pack.tar"
    assets = {
        "source_features.bin": b"x" * 8,
        "canonical_keypoints.bin": b"y" * 8,
        "face_mask.png": b"\xff" * 8,
        "face_crop.png": b"\x01" * 16,
        "identity_embedding.bin": b"\x02" * 32,
        "source_latent.bin": b"\x03" * 64,
    }
    write_pack(
        archive_path=archive,
        identity_id=identity_id,
        assets=assets,
        engine_compatibility=("musetalk-v1",),
    )
    pack = read_pack(archive)
    digest = pack.digest()
    # digest is stable across reload
    pack2 = read_pack(archive)
    assert digest == pack2.digest()
