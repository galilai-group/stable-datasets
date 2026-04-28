"""Image feature codec."""

from __future__ import annotations

import io
from pathlib import Path

from .base import FeatureType


class Image(FeatureType):
    """Image feature stored as raw bytes in Arrow."""

    def __init__(self, encode_format: str = "PNG"):
        self.encode_format = encode_format

    def to_arrow_type(self):
        import pyarrow as pa

        return pa.large_binary()

    def encode(self, value, *, cache_dir: Path | None = None) -> bytes | None:
        return _encode_image_value(value, encode_format=self.encode_format)

    def format(
        self,
        value,
        *,
        format_type: str,
        decode_images: bool = True,
        cache_dir: Path | None = None,
    ):
        if value is None or format_type == "raw" or not decode_images:
            return value

        from PIL import Image as PILImage

        if format_type == "default":
            img = PILImage.open(io.BytesIO(value))
            img.load()
            return img

        import numpy as np

        arr = np.array(PILImage.open(io.BytesIO(value)))
        if format_type == "numpy":
            return arr
        if format_type == "torch":
            import torch

            if arr.ndim == 2:
                arr = arr[:, :, np.newaxis]
            arr = np.ascontiguousarray(arr.transpose(2, 0, 1))
            return torch.from_numpy(arr.astype(np.float32) / 255.0)
        return value

    def __repr__(self) -> str:
        return f"Image(encode_format='{self.encode_format}')"


def _encode_image_value(img, encode_format: str = "PNG") -> bytes | None:
    if img is None:
        return None
    if isinstance(img, bytes):
        return img
    if isinstance(img, str | Path):
        with open(img, "rb") as f:
            return f.read()

    import numpy as np
    from PIL import Image as PILImage

    if isinstance(img, PILImage.Image):
        src = getattr(img, "filename", None)
        if src and Path(src).is_file():
            with open(src, "rb") as f:
                return f.read()
        buf = io.BytesIO()
        fmt = getattr(img, "format", None)
        if fmt is None or img.mode in ("RGBA", "LA", "PA", "P"):
            fmt = "PNG"
        img.save(buf, format=fmt)
        return buf.getvalue()
    if isinstance(img, np.ndarray):
        pil_img = PILImage.fromarray(img)
        buf = io.BytesIO()
        pil_img.save(buf, format=encode_format)
        return buf.getvalue()
    raise TypeError(f"Cannot encode image of type {type(img)}")
