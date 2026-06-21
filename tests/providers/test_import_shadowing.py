"""Regression test for the dynamic-package import helper (Task 1).

LivePortrait upstream sits at ``<repo>/LivePortrait/src/`` and uses
relative imports (``from .config ...``). Our own project has ``src/`` so
``importlib.import_module("src.live_portrait_pipeline")`` would resolve
to our project — wrong shape. The adapter's helper registers the
upstream under a unique name (``liveportrait_upstream``) in
``sys.modules`` to dodge the collision.

This test does NOT require the upstream LivePortrait repo to be cloned:
we fabricate a tiny Python package on ``tmp_path`` and verify the
helper produces a working ``liveportrait_upstream.fabricated`` module
even when a different ``src`` is already in ``sys.modules``.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

from providers.liveportrait.adapter._upstream import _import_upstream_live_portrait


@pytest.fixture
def fabricated_upstream(tmp_path, monkeypatch):
    """Build a fake upstream package on disk and point the env var at it.

    The LivePortrait module shape the helper tries to import
    (``liveportrait_upstream.live_portrait_pipeline``) MUST exist on
    disk, otherwise the helper's :func:`importlib.import_module` raises
    ``ImportError`` and the helper returns ``None``. So we ship a stub
    for that file AND a second fabricated module that the test
    dual-imports to confirm subprocess submodules resolve cleanly.

    Also pretends the project's own ``src`` is already in ``sys.modules``
    so the test exercises the conditional under which the dynamic import
    is necessary.
    """
    upstream_root = tmp_path / "FakeLivePortrait"
    src = upstream_root / "src"
    src.mkdir(parents=True)
    # Make ``src`` a real Python package.
    (src / "__init__.py").write_text("", encoding="utf-8")
    # Stub the *real* module the helper asks for, so its final
    # ``importlib.import_module(module_name)`` call succeeds.
    (src / "live_portrait_pipeline.py").write_text(
        "PIPELINE_FAKE = True\n", encoding="utf-8"
    )
    # A second synthetic submodule the test uses to demonstrate that
    # any subpath under the dynamic package resolves cleanly.
    (src / "fabricated.py").write_text(
        "FABRICATED = 'ok'\n", encoding="utf-8"
    )
    monkeypatch.setenv("HEYAVATAR_LIVE_PORTRAIT_SRC", str(upstream_root))
    # Simulate the project's own ``src`` package being resident — the
    # very condition that would shadow the upstream if loaded naively.
    if "src" not in sys.modules:
        sys.modules["src"] = types.ModuleType("src")
    return src


def _drop(upstream_root: Path) -> None:
    """Remove any cached dynamic-package entries from a previous run."""
    for name in list(sys.modules):
        if name.startswith("liveportrait_upstream"):
            sys.modules.pop(name, None)


def test_import_helper_resolves_when_shadowing_exists(fabricated_upstream, monkeypatch):
    _drop(fabricated_upstream.parent)
    upstream = _import_upstream_live_portrait()
    assert upstream is not None, (
        "Helper should return the registered upstream `live_portrait_pipeline` "
        "submodule even with our own `src` already populated in sys.modules"
    )
    # The helper returns the *submodule* `liveportrait_upstream.live_portrait_pipeline`,
    # not the parent package — confirm both effects with one assertion:
    # (a) `upstream` IS that submodule (its PIPELINE_FAKE stub bleeds through),
    # (b) any other submodule of the dynamic package is reachable through
    #     standard import paths.
    assert upstream.PIPELINE_FAKE is True
    fab = importlib.import_module("liveportrait_upstream.fabricated")
    assert fab.FABRICATED == "ok"


def test_import_helper_caches_when_already_imported(fabricated_upstream):
    _drop(fabricated_upstream.parent)
    first = _import_upstream_live_portrait()
    second = _import_upstream_live_portrait()
    assert first is second
    assert first is not None


def test_import_helper_returns_none_when_src_missing(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "HEYAVATAR_LIVE_PORTRAIT_SRC", str(tmp_path / "no-such-dir")
    )
    _drop(Path("ignored"))
    upstream = _import_upstream_live_portrait()
    assert upstream is None


def test_no_bare_liveportrait_pipeline_import_in_adapters():
    """Fail-loud guard: catch a regression where someone re-introduces
    ``from src.live_portrait_pipeline import X`` in an adapter by
    accident — that would re-trigger the shadowing issue."""
    import re

    bad: list = []
    root = Path("providers")
    if not root.is_dir():
        return
    pattern = re.compile(
        r"(?:from\s+src\.live_portrait_pipeline\s+import"
        r"|import\s+src\.live_portrait_pipeline)"
    )
    for path in root.rglob("*.py"):
        if pattern.search(path.read_text(encoding="utf-8")):
            bad.append(str(path))
    assert not bad, (
        "These modules import `src.live_portrait_pipeline` directly; "
        "they must use the dynamic-package helper in "
        "`providers.liveportrait.adapter._upstream`:\n  - "
        + "\n  - ".join(bad)
    )
