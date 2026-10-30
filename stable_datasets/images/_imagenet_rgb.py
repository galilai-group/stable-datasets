"""Image normalization shared by the ImageNet evaluation-set builders."""

from __future__ import annotations

import io

from PIL import Image as PILImage


def to_rgb_bytes(raw: bytes) -> bytes:
    """Return bytes that decode to an RGB image.

    RGB JPEGs pass through unchanged. Anything else is converted with
    ``convert("RGB")`` and stored as PNG, which is lossless, so pixels match what
    torchvision and timm loaders produce at read time.
    """
    with PILImage.open(io.BytesIO(raw)) as img:
        # Pillow reports some multi-picture camera JPEGs as "MPO".
        if img.format in ("JPEG", "MPO") and img.mode == "RGB":
            return raw
        rgb = img.convert("RGB")
    # The source profile describes the original color space, not RGB, and a large one makes the PNG unreadable.
    rgb.info.pop("icc_profile", None)
    buffer = io.BytesIO()
    rgb.save(buffer, format="PNG")
    return buffer.getvalue()
