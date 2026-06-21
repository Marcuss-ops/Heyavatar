"""Unit tests for ``src.domain.body_template``.

Provides:

* ``BodyTemplate`` direct construction (no file I/O).
* :func:`load_body_template` happy path with synthetic 5-file layout.
* :func:`load_body_template` failure modes (missing files).
* :class:`BodyTemplate` metadata accessors that read
  ``metadata.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.domain.body_template import BodyTemplate, load_body_template
from src.domain.types import IdentityId


# ─────────────────────────────────────────────────────────────────────────────
# synthetic body-template writer
# ─────────────────────────────────────────────────────────────────────────────


def _make_synthetic_template(
    base_dir: Path,
    avatar_id: str,
    gesture_id: str,
    *,
    width: int = 64,
    height: int = 64,
    fps: float = 25.0,
    total_frames: int = 75,
) -> Path:
    """Materialise a 5-file body-template stub under ``base_dir`` and return its dir."""
    import cv2
    import numpy as np

    tmpl_dir = base_dir / avatar_id / gesture_id
    tmpl_dir.mkdir(parents=True, exist_ok=True)

    # Three real cv2 mp4s (solid colour, deterministic frame count).
    for name in ("body.mp4", "face_mask.mp4", "neck_mask.mp4"):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = tmpl_dir / name
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        assert writer.isOpened(), f"failed to open {out_path}"
        frame_kwargs = dict(dtype=np.uint8)
        if "body" in name:
            frame = np.full((height, width, 3), 128, **frame_kwargs)
        elif "face_mask" in name:
            frame = np.full((height, width, 3), 200, **frame_kwargs)
        else:
            frame = np.full((height, width, 3), 80, **frame_kwargs)
        for _ in range(total_frames):
            writer.write(frame)
        writer.release()

    bbox = np.tile(
        np.array([8, 8, width - 8, height - 8], dtype=np.float32),
        (total_frames, 1),
    )
    matrices = np.tile(
        np.eye(4, dtype=np.float32),
        (total_frames, 1, 1),
    )
    np.savez_compressed(
        tmpl_dir / "face_transforms.npz",
        bbox=bbox,
        matrices=matrices,
        landmark_count=total_frames,
    )

    payload = {
        "avatar_id": avatar_id,
        "gesture_id": gesture_id,
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
        "status": "precomputed",
    }
    with open(tmpl_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    return tmpl_dir


# ─────────────────────────────────────────────────────────────────────────────
# BodyTemplate direct construction
# ─────────────────────────────────────────────────────────────────────────────


def test_construction_accepts_paths():
    template = BodyTemplate(
        body_video=Path("/tmp/body.mp4"),
        face_mask=Path("/tmp/face_mask.mp4"),
        neck_mask=Path("/tmp/neck_mask.mp4"),
        face_transforms=Path("/tmp/face_transforms.npz"),
        metadata=Path("/tmp/metadata.json"),
    )
    assert template.body_video.as_posix().endswith("body.mp4")
    assert template.metadata.as_posix().endswith("metadata.json")


def test_construction_is_frozen():
    template = BodyTemplate(
        body_video=Path("/tmp/body.mp4"),
        face_mask=Path("/tmp/face_mask.mp4"),
        neck_mask=Path("/tmp/neck_mask.mp4"),
        face_transforms=Path("/tmp/face_transforms.npz"),
        metadata=Path("/tmp/metadata.json"),
    )
    with pytest.raises((AttributeError, Exception)):
        template.body_video = Path("/tmp/other.mp4")  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# load_body_template happy path
# ─────────────────────────────────────────────────────────────────────────────


def test_load_body_template_resolves_existing_template(tmp_path: Path):
    _make_synthetic_template(tmp_path, "alice", "idle", total_frames=75)

    template = load_body_template("alice", "idle", base_dir=tmp_path)

    assert template.body_video == tmp_path / "alice" / "idle" / "body.mp4"
    assert template.face_mask == tmp_path / "alice" / "idle" / "face_mask.mp4"
    assert template.neck_mask == tmp_path / "alice" / "idle" / "neck_mask.mp4"
    assert template.face_transforms == tmp_path / "alice" / "idle" / "face_transforms.npz"
    assert template.metadata == tmp_path / "alice" / "idle" / "metadata.json"
    # All files exist on disk.
    assert template.body_video.is_file()
    assert template.face_mask.is_file()
    assert template.neck_mask.is_file()
    assert template.face_transforms.is_file()
    assert template.metadata.is_file()


def test_load_body_template_metadata_accessors(tmp_path: Path):
    _make_synthetic_template(
        tmp_path, "alice", "explain_both", width=48, height=36, fps=25.0,
        total_frames=100,
    )
    template = load_body_template("alice", "explain_both", base_dir=tmp_path)

    assert template.avatar_id() == IdentityId("alice")
    assert template.gesture_id() == "explain_both"
    assert template.fps() == 25.0
    assert template.frame_count() == 100
    assert template.width() == 48
    assert template.height() == 36
    assert template.status() == "precomputed"


def test_load_body_template_default_base_dir(tmp_path: Path, monkeypatch):
    """When base_dir is the default 'body_templates' string, the loader treats it as relative CWD."""
    _make_synthetic_template(
        tmp_path / "body_templates", "bob", "idle", total_frames=75
    )
    monkeypatch.chdir(tmp_path)
    template = load_body_template("bob", "idle")
    assert template.body_video.is_file()
    assert template.body_video.name == "body.mp4"


# ─────────────────────────────────────────────────────────────────────────────
# load_body_template failure modes
# ─────────────────────────────────────────────────────────────────────────────


def test_load_body_template_missing_body_video(tmp_path: Path):
    _make_synthetic_template(tmp_path, "alice", "idle", total_frames=75)
    (tmp_path / "alice" / "idle" / "body.mp4").unlink()
    with pytest.raises(FileNotFoundError) as exc:
        load_body_template("alice", "idle", base_dir=tmp_path)
    msg = str(exc.value)
    assert "alice" in msg and "idle" in msg and "body.mp4" in msg


def test_load_body_template_missing_face_mask(tmp_path: Path):
    _make_synthetic_template(tmp_path, "alice", "idle", total_frames=75)
    (tmp_path / "alice" / "idle" / "face_mask.mp4").unlink()
    with pytest.raises(FileNotFoundError) as exc:
        load_body_template("alice", "idle", base_dir=tmp_path)
    assert "face_mask.mp4" in str(exc.value)


def test_load_body_template_missing_neck_mask(tmp_path: Path):
    _make_synthetic_template(tmp_path, "alice", "idle", total_frames=75)
    (tmp_path / "alice" / "idle" / "neck_mask.mp4").unlink()
    with pytest.raises(FileNotFoundError) as exc:
        load_body_template("alice", "idle", base_dir=tmp_path)
    assert "neck_mask.mp4" in str(exc.value)


def test_load_body_template_missing_face_transforms(tmp_path: Path):
    _make_synthetic_template(tmp_path, "alice", "idle", total_frames=75)
    (tmp_path / "alice" / "idle" / "face_transforms.npz").unlink()
    with pytest.raises(FileNotFoundError) as exc:
        load_body_template("alice", "idle", base_dir=tmp_path)
    assert "face_transforms.npz" in str(exc.value)


def test_load_body_template_missing_metadata(tmp_path: Path):
    _make_synthetic_template(tmp_path, "alice", "idle", total_frames=75)
    (tmp_path / "alice" / "idle" / "metadata.json").unlink()
    with pytest.raises(FileNotFoundError) as exc:
        load_body_template("alice", "idle", base_dir=tmp_path)
    assert "metadata.json" in str(exc.value)


def test_load_body_template_missing_all_files_lists_them(tmp_path: Path):
    # No template at all → all 5 paths listed in error.
    with pytest.raises(FileNotFoundError) as exc:
        load_body_template("ghost", "idle", base_dir=tmp_path)
    msg = str(exc.value)
    for filename in (
        "body.mp4", "face_mask.mp4", "neck_mask.mp4",
        "face_transforms.npz", "metadata.json",
    ):
        assert filename in msg, f"{filename} missing from error message"
