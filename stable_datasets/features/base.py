"""Core feature descriptors shared across modalities."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa


class FeatureType:
    """Base class for feature type descriptors."""

    def to_arrow_type(self) -> pa.DataType:
        raise NotImplementedError

    def arrow_metadata(self) -> dict[bytes, bytes]:
        return {b"stable_datasets.feature": type(self).__name__.encode()}

    def encode(self, value, *, cache_dir: Path | None = None):
        if hasattr(value, "item"):
            return value.item()
        return value

    def format(
        self,
        value,
        *,
        format_type: str,
        decode_images: bool = True,
        cache_dir: Path | None = None,
    ):
        return value

    def fingerprint_data(self) -> str:
        return repr(self)


class Value(FeatureType):
    """Scalar value type. Maps dtype strings to PyArrow types."""

    _DTYPE_MAP: dict[str, pa.DataType] = {
        "int8": pa.int8(),
        "int16": pa.int16(),
        "int32": pa.int32(),
        "int64": pa.int64(),
        "uint8": pa.uint8(),
        "uint16": pa.uint16(),
        "uint32": pa.uint32(),
        "uint64": pa.uint64(),
        "float16": pa.float16(),
        "float32": pa.float32(),
        "float64": pa.float64(),
        "bool": pa.bool_(),
        "string": pa.string(),
        "binary": pa.binary(),
    }

    def __init__(self, dtype: str):
        if dtype not in self._DTYPE_MAP:
            raise ValueError(f"Unknown dtype '{dtype}'. Supported: {list(self._DTYPE_MAP)}")
        self.dtype = dtype

    def to_arrow_type(self) -> pa.DataType:
        return self._DTYPE_MAP[self.dtype]

    def format(
        self,
        value,
        *,
        format_type: str,
        decode_images: bool = True,
        cache_dir: Path | None = None,
    ):
        if format_type == "torch" and isinstance(value, int | float | bool):
            import torch

            return torch.tensor(value)
        return value

    def __repr__(self) -> str:
        return f"Value('{self.dtype}')"


class ClassLabel(FeatureType):
    """Categorical label with name-to-int mapping."""

    def __init__(self, names: list[str] | None = None, num_classes: int | None = None):
        if names is not None:
            self.names: list[str] = list(names)
            self.num_classes: int = len(names)
        elif num_classes is not None:
            self.num_classes = num_classes
            self.names = [str(i) for i in range(num_classes)]
        else:
            raise ValueError("ClassLabel requires either 'names' or 'num_classes'")
        self._str2int: dict[str, int] = {n: i for i, n in enumerate(self.names)}
        self._int2str: dict[int, str] = dict(enumerate(self.names))

    def str2int(self, name: str) -> int:
        return self._str2int[name]

    def int2str(self, idx: int) -> str:
        return self._int2str[idx]

    def to_arrow_type(self) -> pa.DataType:
        return pa.int64()

    def encode(self, value, *, cache_dir: Path | None = None):
        if isinstance(value, str):
            return self.str2int(value)
        if hasattr(value, "item"):
            return value.item()
        return value

    def format(
        self,
        value,
        *,
        format_type: str,
        decode_images: bool = True,
        cache_dir: Path | None = None,
    ):
        if format_type == "torch" and isinstance(value, int | float | bool):
            import torch

            return torch.tensor(value)
        return value

    def __repr__(self) -> str:
        if len(self.names) <= 5:
            return f"ClassLabel(names={self.names})"
        return f"ClassLabel(num_classes={self.num_classes})"


class Sequence(FeatureType):
    """Variable-length list of a sub-feature."""

    def __init__(self, feature: FeatureType):
        self.feature = feature

    def to_arrow_type(self) -> pa.DataType:
        return pa.list_(self.feature.to_arrow_type())

    def encode(self, value, *, cache_dir: Path | None = None):
        if value is None:
            return None
        if hasattr(value, "tolist"):
            return value.tolist()
        return list(value)

    def __repr__(self) -> str:
        return f"Sequence({self.feature!r})"
