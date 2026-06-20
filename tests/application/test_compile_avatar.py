"""Avatar compilation use case tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from providers import get_provider
from src.application.compile_avatar import AvatarCompiler
from src.domain.enums import EngineId
from src.domain.types import IdentitySpec
from tests._fixtures import PNG_1X1 as _PNG_1x1


def test_compile_writes_pack_and_handle(workdir, tmp_path):
    source = tmp_path / "alice.png"
    source.write_bytes(_PNG_1x1)
    engine = get_provider(EngineId.MUSE_TALK)
    engine.load()
    try:
        compiler = AvatarCompiler(engine=engine, pack_root=workdir / "packs")
        handle = compiler.compile(
            IdentitySpec(source_image=source, display_name="Alice", language_hint="it-IT"),
        )
        assert handle.pack_path.is_file()
        assert handle.identity_id.startswith("id-")
        # Identity is deterministic — same inputs produce same id.
        handle_again = compiler.compile(
            IdentitySpec(source_image=source, display_name="Alice", language_hint="it-IT"),
        )
        assert handle.identity_id == handle_again.identity_id
    finally:
        engine.unload()
