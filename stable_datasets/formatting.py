"""Formatter classes for converting Arrow-native values to user-facing types.

Three-layer separation: ArrowBackend -> Formatter -> StableDataset.
The backend never imports PIL or torch. The formatter never touches Arrow files.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image as PILImage

from .schema import Array3D, Features, Image


class Formatter:
    """Base formatter. Subclasses convert Arrow-native values to user-facing types."""

    def __init__(self, features: Features, decode_images: bool = True):
        self.features = features
        self.decode_images = decode_images
        self._image_cols = [n for n, f in features.items() if isinstance(f, Image)]
        self._array3d_cols = [n for n, f in features.items() if isinstance(f, Array3D)]

    def format_row(self, row: dict) -> dict:
        """Format a single row dict (from backend.get_row)."""
        raise NotImplementedError

    def format_batch(self, table) -> list[dict]:
        """Format a batch (from backend.take). Returns list of row dicts.

        Default implementation calls ``to_pydict()`` then formats each row.
        Subclasses can override for more efficient batch conversion.
        """
        batch_dict = table.to_pydict()
        n = table.num_rows
        rows = []
        for i in range(n):
            row = {k: v[i] for k, v in batch_dict.items()}
            rows.append(self.format_row(row))
        return rows


class PythonFormatter(Formatter):
    """Default format: Image -> PIL, Array3D -> numpy, scalars -> Python native."""

    def format_row(self, row: dict) -> dict:
        result = dict(row)
        if self.decode_images:
            for col in self._image_cols:
                val = result[col]
                if val is not None:
                    img = PILImage.open(io.BytesIO(val))
                    img.load()
                    result[col] = img
        for col in self._array3d_cols:
            val = result[col]
            if val is not None:
                feat = self.features[col]
                result[col] = np.frombuffer(val, dtype=feat.dtype).reshape(feat.shape)
        return result


class RawFormatter(Formatter):
    """Raw format: all values as-is from Arrow (bytes for images, bytes for Array3D)."""

    def __init__(self, features: Features, decode_images: bool = False):
        super().__init__(features, decode_images=False)

    def format_row(self, row: dict) -> dict:
        return row


class TorchFormatter(Formatter):
    """Torch format: Image -> CHW float32 tensor, Array3D -> float32 tensor, scalars -> tensors."""

    def format_row(self, row: dict) -> dict:
        import torch

        result = {}
        for col_name, value in row.items():
            if value is None:
                result[col_name] = None
                continue
            feat = self.features.get(col_name)
            if isinstance(feat, Image):
                if self.decode_images:
                    img = PILImage.open(io.BytesIO(value))
                    arr = np.array(img)
                    if arr.ndim == 2:
                        arr = arr[:, :, np.newaxis]
                    arr = np.ascontiguousarray(arr.transpose(2, 0, 1))
                    result[col_name] = torch.from_numpy(arr.astype(np.float32) / 255.0)
                else:
                    result[col_name] = value
            elif isinstance(feat, Array3D):
                result[col_name] = torch.from_numpy(
                    np.frombuffer(value, dtype=feat.dtype).reshape(feat.shape).astype(np.float32)
                )
            elif isinstance(value, (int, float)):
                result[col_name] = torch.tensor(value)
            else:
                result[col_name] = value
        return result


class NumpyFormatter(Formatter):
    """Numpy format: Image -> HWC numpy array, rest as-is."""

    def format_row(self, row: dict) -> dict:
        result = dict(row)
        if self.decode_images:
            for col in self._image_cols:
                val = result[col]
                if val is not None:
                    result[col] = np.array(PILImage.open(io.BytesIO(val)))
        for col in self._array3d_cols:
            val = result[col]
            if val is not None:
                feat = self.features[col]
                result[col] = np.frombuffer(val, dtype=feat.dtype).reshape(feat.shape)
        return result


def get_formatter(
    format_type: str | None,
    features: Features,
    decode_images: bool = True,
) -> Formatter:
    """Factory for formatter instances."""
    if format_type is None:
        return PythonFormatter(features, decode_images=decode_images)
    if format_type == "raw":
        return RawFormatter(features, decode_images=False)
    if format_type == "torch":
        return TorchFormatter(features, decode_images=decode_images)
    if format_type == "numpy":
        return NumpyFormatter(features, decode_images=decode_images)
    raise ValueError(f"Unknown format type: {format_type!r}")
