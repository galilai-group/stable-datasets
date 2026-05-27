"""Array-based feature codecs."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from .base import FeatureType


class _FixedShapeArray(FeatureType):
    """Fixed-shape array stored as flat bytes."""

    _ndim: int

    def __init__(self, shape: tuple, dtype: str = "uint8"):
        if len(shape) != self._ndim:
            raise ValueError(f"{type(self).__name__} requires a {self._ndim}-D shape; got {shape}.")
        self.shape = tuple(shape)
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
        return f"{type(self).__name__}(shape={self.shape}, dtype='{self.dtype}')"


class Array3D(_FixedShapeArray):
    """Fixed-shape 3D array stored as flat bytes."""

    _ndim = 3


class Array4D(_FixedShapeArray):
    """Fixed-shape 4D array stored as flat bytes."""

    _ndim = 4
