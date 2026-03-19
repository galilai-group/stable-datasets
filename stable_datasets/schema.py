"""Feature and metadata schema definitions.

Each feature type maps itself to a PyArrow type for Arrow IPC serialization.
"""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa


class Version:
    """Semantic version string (``major.minor.patch``)."""

    def __init__(self, version_str: str):
        parts = version_str.split(".")
        if len(parts) != 3:
            raise ValueError(f"Version string must be 'major.minor.patch', got '{version_str}'")
        self.major, self.minor, self.patch = int(parts[0]), int(parts[1]), int(parts[2])
        self._str = version_str

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        return f"Version('{self._str}')"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Version):
            return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch))


class FeatureType:
    """Base class for feature type descriptors."""

    def to_arrow_type(self) -> pa.DataType:
        raise NotImplementedError


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

    def __repr__(self) -> str:
        return f"Value('{self.dtype}')"


class ClassLabel(FeatureType):
    """Categorical label with name-to-int mapping.

    Preserves the ``.names``, ``.num_classes``, ``.str2int()``, ``.int2str()``
    API that downstream code relies on.
    """

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

    def __repr__(self) -> str:
        if len(self.names) <= 5:
            return f"ClassLabel(names={self.names})"
        return f"ClassLabel(num_classes={self.num_classes})"


class Image(FeatureType):
    """Image feature. Stored as raw bytes (PNG-encoded) in Arrow."""

    def to_arrow_type(self) -> pa.DataType:
        return pa.binary()

    def __repr__(self) -> str:
        return "Image()"


class Video(FeatureType):
    """Video feature. Stored as file path string in Arrow (metadata-only).

    Video bytes are never inlined into the Arrow cache.  The path points to
    the source media file; decoding happens lazily at access time.
    """

    def to_arrow_type(self) -> pa.DataType:
        return pa.string()

    def __repr__(self) -> str:
        return "Video()"


class Sequence(FeatureType):
    """Variable-length list of a sub-feature."""

    def __init__(self, feature: FeatureType):
        self.feature = feature

    def to_arrow_type(self) -> pa.DataType:
        return pa.list_(self.feature.to_arrow_type())

    def __repr__(self) -> str:
        return f"Sequence({self.feature!r})"


class Array3D(FeatureType):
    """Fixed-shape 3D array (e.g. 3D medical volumes). Stored as flat bytes."""

    def __init__(self, shape: tuple, dtype: str = "uint8"):
        self.shape = shape
        self.dtype = dtype

    def to_arrow_type(self) -> pa.DataType:
        return pa.binary()

    def __repr__(self) -> str:
        return f"Array3D(shape={self.shape}, dtype='{self.dtype}')"


class Features(dict):
    """Ordered dict of ``field_name -> FeatureType``.

    Generates a PyArrow schema via ``.to_arrow_schema()``.
    """

    def to_arrow_schema(self) -> pa.schema:
        fields = []
        for name, feat in self.items():
            if not isinstance(feat, FeatureType):
                raise TypeError(f"Feature '{name}' must be a FeatureType, got {type(feat).__name__}")
            fields.append(pa.field(name, feat.to_arrow_type()))
        return pa.schema(fields)


@dataclass
class DatasetInfo:
    """Metadata container for a dataset (description, features, citation, etc.)."""

    features: Features
    description: str = ""
    supervised_keys: tuple | None = None
    homepage: str = ""
    citation: str = ""
    license: str = ""
    config_name: str = ""


@dataclass
class BuilderConfig:
    """Base config for multi-variant datasets."""

    name: str = "default"
    version: Version | None = None
    description: str = ""
