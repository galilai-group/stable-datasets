"""Array-based feature codecs."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from .base import FeatureType


class Array3D(FeatureType):
    """Fixed-shape 3D array stored as flat bytes."""

    def __init__(self, shape: tuple, dtype: str = "uint8"):
        self.shape = shape
        self.dtype = dtype

    def to_arrow_type(self) -> pa.DataType:
        return pa.large_binary()

    def encode(self, value, *, cache_dir: Path | None = None) -> bytes | None:
        if value is None:
            return None
        import numpy as np

        arr = np.asarray(value, dtype=self.dtype)
        return arr.tobytes()

    def format(
        self,
        value,
        *,
        format_type: str,
        decode_images: bool = True,
        cache_dir: Path | None = None,
    ):
        if value is None or format_type == "raw":
            return value
        import numpy as np

        arr = np.frombuffer(value, dtype=self.dtype).reshape(self.shape)
        if format_type == "torch":
            import torch

            return torch.from_numpy(arr.astype(np.float32))
        return arr

    def __repr__(self) -> str:
        return f"Array3D(shape={self.shape}, dtype='{self.dtype}')"
