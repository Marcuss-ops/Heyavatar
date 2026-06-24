"""Avatar image preprocessing helpers.

These helpers keep one-time source-image cleanup out of the hot render
path. They are used to build reusable avatar base assets from green-screen
portraits before identity compilation.
"""

from __future__ import annotations

import io
import base64
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_GREEN_LOWER = np.array([35, 40, 40], dtype=np.uint8)
_GREEN_UPPER = np.array([90, 255, 255], dtype=np.uint8)


def build_green_screen_cutout(image_rgb: np.ndarray) -> tuple[np.ndarray, float]:
    """Return an RGBA cutout and the fraction of pixels treated as green.

    The input must be an ``HxWx3`` RGB array. The output keeps the subject
    opaque while turning the detected green screen transparent.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got {image_rgb.shape!r}")

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    green_mask = cv2.inRange(hsv, _GREEN_LOWER, _GREEN_UPPER)
    green_mask = cv2.medianBlur(green_mask, 5)
    green_mask = cv2.GaussianBlur(green_mask, (0, 0), 2.0)

    alpha = 255 - green_mask
    alpha = np.clip(alpha, 0, 255).astype(np.uint8)
    rgba = np.dstack([image_rgb.astype(np.uint8), alpha])

    green_ratio = float(np.count_nonzero(green_mask > 32)) / float(green_mask.size)
    return rgba, green_ratio


def rgba_png_bytes(rgba_image: np.ndarray) -> bytes:
    """Serialize an RGBA image array to PNG bytes."""
    if rgba_image.ndim != 3 or rgba_image.shape[2] != 4:
        raise ValueError(f"Expected RGBA image with 4 channels, got {rgba_image.shape!r}")
    buffer = io.BytesIO()
    Image.fromarray(rgba_image, mode="RGBA").save(buffer, format="PNG")
    return buffer.getvalue()


def write_green_screen_cutout_svg(source_image: Path, output_path: Path) -> tuple[Path, float]:
    """Write an SVG wrapper that embeds the transparent cutout image.

    The SVG stores the raster cutout as an embedded PNG data URI, so we
    keep the alpha matte without creating another standalone PNG asset.
    """
    image = Image.open(source_image).convert("RGB")
    rgba, green_ratio = build_green_screen_cutout(np.asarray(image, dtype=np.uint8))
    png_bytes = rgba_png_bytes(rgba)
    data_uri = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
    width, height = rgba.shape[1], rgba.shape[0]
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <image href="{data_uri}" width="{width}" height="{height}" preserveAspectRatio="none" />
</svg>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")
    return output_path, green_ratio


def build_green_screen_cutout_png(image_rgb: np.ndarray) -> tuple[bytes, float]:
    """Return PNG bytes for a transparent cutout plus the green-screen ratio."""
    rgba, green_ratio = build_green_screen_cutout(image_rgb)
    return rgba_png_bytes(rgba), green_ratio


def write_green_screen_cutout(source_image: Path, output_path: Path) -> tuple[Path, float]:
    """Write a transparent cutout PNG next to the source image."""
    image = Image.open(source_image).convert("RGB")
    rgba, green_ratio = build_green_screen_cutout(np.asarray(image, dtype=np.uint8))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(output_path)
    return output_path, green_ratio


def load_avatar_source_image(source_image: Path) -> Image.Image:
    """Load a raster or SVG-wrapped avatar source image.

    SVG assets are expected to use the wrapper format written by
    :func:`write_green_screen_cutout_svg`, which embeds a PNG data URI.
    """
    if source_image.suffix.lower() != ".svg":
        return Image.open(source_image).convert("RGB")

    root = ET.fromstring(source_image.read_text(encoding="utf-8"))
    image = root.find("{http://www.w3.org/2000/svg}image")
    if image is None:
        raise ValueError(f"SVG avatar source has no embedded image: {source_image}")
    href = image.attrib.get("{http://www.w3.org/1999/xlink}href") or image.attrib.get("href")
    if not href or not href.startswith("data:image/png;base64,"):
        raise ValueError(f"SVG avatar source does not contain an embedded PNG data URI: {source_image}")
    png_bytes = base64.b64decode(href.split(",", 1)[1])
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


__all__ = [
    "build_green_screen_cutout",
    "build_green_screen_cutout_png",
    "load_avatar_source_image",
    "rgba_png_bytes",
    "write_green_screen_cutout",
    "write_green_screen_cutout_svg",
]
