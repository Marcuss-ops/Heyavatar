"""Contract test for the AvatarEngine ABC.

Verifies that every registered provider implements the contract
correctly: load → prepare_identity (dict of bytes) → render_chunk
(RenderChunkResult with required fields).
"""

from __future__ import annotations

import shutil

import pytest

from providers import get_provider, PROVIDERS
from src.core.config import get_settings
from src.domain.enums import EngineId
from src.domain.avatar_pack import write_pack, read_pack
from src.domain.types import (
    AvatarIdentityHandle,
    IdentityId,
    IdentitySpec,
    RenderChunkRequest,
)
from tests._fixtures import PNG_1X1 as _PNG_1x1


requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="mock-mode provider tests shell out to ffmpeg",
)


@requires_ffmpeg
@pytest.mark.parametrize("engine_id", list(PROVIDERS))
def test_provider_passes_contract(workdir, engine_id, tmp_path):
    settings = get_settings()
    assert settings.mock_engine, "tests must run in mock mode"

    source = tmp_path / "face.png"
    source.write_bytes(_PNG_1x1)
    adapter = get_provider(engine_id)
    assert adapter.engine_id == engine_id

    adapter.load()
    try:
        assets = adapter.prepare_identity(source)
        assert isinstance(assets, dict)
        assert "identity_embedding.bin" in assets
        # Round-trip through a pack to mimic production flow.
        pack_path = workdir / "packs" / f"{identity_id_for(adapter, source)}.tar"
        pack_path.parent.mkdir(parents=True, exist_ok=True)
        pack = write_pack(
            archive_path=pack_path,
            identity_id=identity_id_for(adapter, source),
            assets=assets,
            engine_compatibility=(adapter.engine_id.value,),
        )
        handle = AvatarIdentityHandle(
            identity_id=pack.manifest.identity_id,
            pack_path=pack_path,
            pack_digest=pack.digest(),
            prepared_at=pack.manifest.created_at,
        )
        # Render one chunk.
        req = RenderChunkRequest(
            job_id="job-test",
            audio_window=(0.0, 1.0),
            audio_path=tmp_path / "speech.wav",
            fps=25,
            resolution=(512, 512),
            chunk_index=0,
        )
        (tmp_path / "speech.wav").write_bytes(b"RIFF0000WAVE")
        result = adapter.render_chunk(req, handle)
        assert result.chunk_index == 0
        assert result.frames_rendered > 0
        assert result.gpu_seconds >= 0
        # Health must respond.
        h = adapter.health()
        assert h.engine_id == adapter.engine_id
        assert h.mock_mode is True
    finally:
        adapter.unload()


def identity_id_for(provider, source) -> IdentityId:
    import hashlib
    digest = hashlib.sha256()
    digest.update(provider.engine_id.value.encode("utf-8"))
    digest.update(str(source).encode("utf-8"))
    return IdentityId(f"id-{digest.hexdigest()[:12]}")
