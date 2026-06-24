from __future__ import annotations

import numpy as np
from PIL import Image

from src.application.avatar_image import (
    build_green_screen_cutout,
    build_green_screen_cutout_png,
    load_avatar_source_image,
    write_green_screen_cutout_svg,
)


def test_build_green_screen_cutout_creates_transparent_background() -> None:
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    image[:, :] = (0, 255, 0)
    image[24:72, 28:68] = (220, 40, 30)

    rgba, green_ratio = build_green_screen_cutout(image)

    assert rgba.shape == (96, 96, 4)
    assert green_ratio > 0.5
    assert rgba[8, 8, 3] < 16
    assert rgba[48, 48, 3] > 200


def test_build_green_screen_cutout_png_returns_valid_png_bytes() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :] = (0, 255, 0)
    image[16:48, 16:48] = (240, 240, 240)

    png_bytes, green_ratio = build_green_screen_cutout_png(image)

    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    assert green_ratio > 0.5


def test_svg_cutout_loader_round_trips_embedded_image(tmp_path) -> None:
    source = tmp_path / "source.png"
    svg = tmp_path / "cutout.svg"
    image = Image.new("RGB", (32, 32), (0, 255, 0))
    image.paste(Image.new("RGB", (8, 8), (240, 200, 180)), (12, 12))
    image.save(source)

    write_green_screen_cutout_svg(source, svg)
    loaded = load_avatar_source_image(svg)

    assert loaded.mode == "RGB"
    assert loaded.size == (32, 32)
