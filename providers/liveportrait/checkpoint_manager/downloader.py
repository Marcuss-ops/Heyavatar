"""Download backends and the smart-selector dispatcher.

* :func:`_parse_hf_url` — extract ``(repo_id, revision, filename)`` from
  a HuggingFace Hub URL.
* :func:`_hf_download_to` — HF Hub ``hf_hub_download`` (preferred;
  handles LFS, integrity checks, resumable downloads, progress bars).
* :func:`_urllib_download_to` — plain HTTP fallback with optional
  ``tqdm`` progress bar.
* :func:`_download_to_smart` — entry point used by the manager that
  inspects the URL and dispatches to the appropriate backend.

HF Hub URLs MUST use the HF Python API — plain urllib on a HF
``resolve/main`` URL downloads the Git LFS pointer (a tiny text
blob), not the actual model weights.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

from src.core.logging import get_logger


_HF_URL_RE = re.compile(
    r"^https://huggingface\.co/([^/]+/[^/]+)/resolve/([^/]+)/(.+)$"
)


def _parse_hf_url(url: str) -> Optional[Tuple[str, str, str]]:
    """Extract (repo_id, revision, filename) from a HuggingFace Hub URL.

    Returns None if the URL does not match the HF Hub ``resolve`` pattern.
    """
    m = _HF_URL_RE.match(url)
    if m is None:
        return None
    return m.group(1), m.group(2), m.group(3)


def _hf_download_to(repo_id: str, filename: str, revision: str, dest: Path, cache_dir: Path) -> None:
    """Download a single file from HuggingFace Hub via ``huggingface_hub``.

    Uses ``hf_hub_download()`` which handles Git LFS, built-in integrity
    checks, resumable downloads, and progress bars. The file is placed
    at ``dest`` (not inside the HF cache).

    Caller must ensure ``huggingface_hub`` is importable before calling.
    """
    log = get_logger(__name__)
    from huggingface_hub import hf_hub_download

    log.info(
        "Downloading via HuggingFace Hub: %s :: %s @ %s → %s",
        repo_id, filename, revision, dest,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    # hf_hub_download downloads to the HF cache, then we copy to our
    # managed root so we control the file layout.
    try:
        cached = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=str(cache_dir.parent / "_hf_cache"),
        )
    except Exception as exc:
        log.error("hf_hub_download failed for %s/%s: %s", repo_id, filename, exc)
        raise RuntimeError(
            f"Failed to download {repo_id}/{filename} via HuggingFace Hub: {exc}"
        ) from exc

    # Copy from HF cache to our managed location.
    cached_path = Path(cached)
    if cached_path.resolve() != dest.resolve():
        fd, tmp_name = tempfile.mkstemp(prefix=dest.name + ".", dir=dest.parent)
        try:
            os.close(fd)
            shutil.copy2(cached_path, tmp_name)
            os.replace(tmp_name, dest)
        except OSError as exc:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise RuntimeError(
                f"Failed to copy checkpoint from HF cache to {dest}: {exc}"
            ) from exc
    else:
        # Same path — no copy needed.
        pass

    log.info("Downloaded checkpoint %s (%d bytes)", dest.name, dest.stat().st_size)


def _urllib_download_to(url: str, dest: Path, *, show_progress: bool = True) -> None:
    """Stream ``url`` to ``dest`` via urllib with optional progress bar."""
    log = get_logger(__name__)
    log.info("Downloading checkpoint via HTTP %s → %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=dest.name + ".", dir=dest.parent)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "heyavatar/0.2"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            total = resp.headers.get("Content-Length")
            total_bytes = int(total) if total else None
            with os.fdopen(fd, "wb") as tmp_fh:
                if show_progress and total_bytes:
                    # Try tqdm for a progress bar; degrade silently.
                    try:
                        from tqdm import tqdm as _tqdm
                        with _tqdm(
                            total=total_bytes, unit="B", unit_scale=True,
                            desc=dest.name, miniters=1,
                        ) as pbar:
                            while True:
                                chunk = resp.read(1 << 16)
                                if not chunk:
                                    break
                                tmp_fh.write(chunk)
                                pbar.update(len(chunk))
                    except ImportError:
                        shutil.copyfileobj(resp, tmp_fh, length=1 << 16)
                else:
                    shutil.copyfileobj(resp, tmp_fh, length=1 << 16)
        os.replace(tmp_name, dest)
    except (urllib.error.URLError, OSError) as exc:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        log.error("Failed to download %s: %s", url, exc)
        raise RuntimeError(
            f"Could not download checkpoint at {url}: {exc}"
        ) from exc

    log.info("Downloaded checkpoint %s (%d bytes)", dest.name, dest.stat().st_size)


def _download_to_smart(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest``, preferring HuggingFace Hub API.

    1. If the URL is a HuggingFace ``resolve`` path: require
       ``huggingface_hub`` and use ``_hf_download_to()``.
    2. Otherwise fall back to ``_urllib_download_to()`` with optional
       ``tqdm`` progress bar.

    HF Hub URLs MUST use the HF Python API — plain urllib on a HF
    ``resolve/main`` URL downloads the Git LFS pointer (a tiny text
    blob), not the actual model weights.
    """
    parsed = _parse_hf_url(url)
    if parsed is not None:
        repo_id, revision, filename = parsed
        # Require huggingface_hub for HF URLs — urllib fallback would
        # download the LFS pointer, not the weights.
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "huggingface_hub is not installed, but checkpoint URL "
                f"is a HuggingFace Hub path: {url}. Install it with "
                "`pip install huggingface_hub`, or set "
                "HEYAVATAR_LIVE_PORTRAIT_MOCK_DOWNLOAD=1 to skip downloads."
            ) from None
        _hf_download_to(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            dest=dest,
            cache_dir=dest.parent,
        )
        return
    _urllib_download_to(url, dest)


# ── backwards-compat alias (tests may monkey-patch) ────────────────

_download_to = _urllib_download_to
