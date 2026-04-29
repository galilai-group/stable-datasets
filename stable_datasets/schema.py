"""Feature and metadata schema definitions.

Each feature type maps itself to a PyArrow type for Arrow IPC serialization.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping
from collections.abc import Sequence as ABCSequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, NewType, Protocol

import pyarrow as pa

from .features import Array3D, ClassLabel, FeatureType, Image, Sequence, Value, Video, VideoRef


__all__ = [
    "Array3D",
    "BuilderConfig",
    "ClassLabel",
    "DatasetInfo",
    "DatasetSource",
    "DownloadInfo",
    "FeatureType",
    "Features",
    "Image",
    "Sequence",
    "URL",
    "Value",
    "Version",
    "Video",
    "VideoDecodeConfig",
    "VideoDecodeFn",
    "VideoDecodeFnBatched",
    "VideoRef",
    "collect_dataset_citations",
]


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


@dataclass
class DownloadInfo:
    """Download source metadata for one raw asset.

    ``url`` is attempted first. Any ``fallbacks`` are tried in order if the
    primary URL fails.
    """

    url: str
    fallbacks: list[str] = field(default_factory=list)
    checksum: str | None = None
    filename: str | None = None

    def __post_init__(self):
        if not isinstance(self.url, str) or not self.url:
            raise TypeError("DownloadInfo.url must be a non-empty string.")
        if not isinstance(self.fallbacks, list) or not all(isinstance(url, str) and url for url in self.fallbacks):
            raise TypeError("DownloadInfo.fallbacks must be a list of non-empty strings.")
        if self.checksum is not None and not isinstance(self.checksum, str):
            raise TypeError("DownloadInfo.checksum must be a string when provided.")
        if self.filename is not None and not isinstance(self.filename, str):
            raise TypeError("DownloadInfo.filename must be a string when provided.")

    def all_urls(self) -> list[str]:
        return [self.url, *self.fallbacks]


URL = NewType("URL", str)


@dataclass(frozen=True)
class DatasetSource(Mapping[str, object]):
    """Typed source and download metadata for one dataset builder."""

    homepage: URL | str
    assets: dict[str, DownloadInfo | str]
    citation: str
    license: str = ""
    checksums: dict[str, str] | None = None

    def __post_init__(self):
        if not isinstance(self.homepage, str) or not self.homepage:
            raise TypeError("DatasetSource.homepage must be a non-empty string.")
        if not isinstance(self.citation, str) or not self.citation:
            raise TypeError("DatasetSource.citation must be a non-empty string.")
        if not isinstance(self.license, str):
            raise TypeError("DatasetSource.license must be a string.")
        if not isinstance(self.assets, Mapping):
            raise TypeError("DatasetSource.assets must be a mapping.")

        normalized_assets = {}
        for key, value in self.assets.items():
            if not isinstance(key, str) or not key:
                raise TypeError("DatasetSource asset keys must be non-empty strings.")
            if isinstance(value, str):
                normalized_assets[key] = DownloadInfo(url=value)
            elif isinstance(value, DownloadInfo):
                normalized_assets[key] = value
            else:
                raise TypeError(
                    f"DatasetSource.assets['{key}'] must be a URL string or DownloadInfo, got {type(value).__name__}."
                )

        normalized_checksums = None
        if self.checksums is not None:
            if not isinstance(self.checksums, Mapping):
                raise TypeError("DatasetSource.checksums must be a mapping when provided.")
            normalized_checksums = {}
            for key, value in self.checksums.items():
                if not isinstance(key, str) or not key:
                    raise TypeError("DatasetSource.checksums keys must be non-empty strings.")
                if not isinstance(value, str) or not value:
                    raise TypeError("DatasetSource.checksums values must be non-empty strings.")
                normalized_checksums[key] = value

        object.__setattr__(self, "homepage", str(self.homepage))
        object.__setattr__(self, "assets", MappingProxyType(normalized_assets))
        object.__setattr__(
            self,
            "checksums",
            None if normalized_checksums is None else MappingProxyType(normalized_checksums),
        )

    def __getitem__(self, key: str) -> object:
        if key == "homepage":
            return self.homepage
        if key == "assets":
            return self.assets
        if key == "citation":
            return self.citation
        if key == "license":
            return self.license
        if key == "checksums":
            return self.checksums
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        yield "homepage"
        yield "assets"
        yield "citation"
        if self.license:
            yield "license"
        if self.checksums is not None:
            yield "checksums"

    def __len__(self) -> int:
        return 3 + int(bool(self.license)) + int(self.checksums is not None)

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def collect_dataset_citations(sources: Iterable[DatasetSource | Mapping[str, object]]) -> list[str]:
    """Collect unique citation strings in stable first-seen order."""
    citations = []
    seen = set()
    for source in sources:
        citation = source["citation"] if isinstance(source, Mapping) else source.citation
        if citation not in seen:
            seen.add(citation)
            citations.append(citation)
    return citations


class VideoDecodeFn(Protocol):
    """Per-sample video decode callback."""

    def __call__(
        self,
        ref: VideoRef,
        config: VideoDecodeConfig,
        *,
        row: Mapping[str, Any] | None = None,
        sample_index: int | None = None,
    ) -> Any: ...


class VideoDecodeFnBatched(Protocol):
    """Batched video decode callback used by ``StableDataset.__getitems__``."""

    def __call__(
        self,
        refs: ABCSequence[VideoRef],
        config: VideoDecodeConfig,
        *,
        rows: ABCSequence[Mapping[str, Any]] | None = None,
        sample_indices: ABCSequence[int] | None = None,
    ) -> ABCSequence[Any]: ...


@dataclass(frozen=True)
class VideoDecodeConfig:
    """Read-time video decode configuration.

    This is retrieval policy only: it does not affect cache construction,
    cache fingerprints, or the persisted schema.
    """

    num_frames: int
    column: str = "video"
    sampling: Literal["uniform", "random", "center", "start"] = "uniform"
    frame_stride: int = 1
    decoder: Literal["torchcodec", "decord", "cv2"] = "torchcodec"
    output: Literal["torch", "numpy"] = "torch"
    layout: Literal["TCHW", "CTHW", "THWC"] = "TCHW"
    dtype: Literal["float32", "uint8"] = "float32"
    scale: Literal["zero_one", "none"] = "zero_one"
    resize: int | tuple[int, int] | None = None
    crop: Literal["none", "center", "random"] = "none"
    pad: Literal["error", "repeat_last", "loop"] = "error"
    seed: int | None = None
    decode_fn: VideoDecodeFn | None = None
    decode_fn_batched: VideoDecodeFnBatched | None = None

    def __post_init__(self):
        if not isinstance(self.num_frames, int) or isinstance(self.num_frames, bool):
            raise TypeError("num_frames must be an int.")
        if self.num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {self.num_frames}")
        if not isinstance(self.frame_stride, int) or isinstance(self.frame_stride, bool):
            raise TypeError("frame_stride must be an int.")
        if self.frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {self.frame_stride}")
        if not isinstance(self.column, str) or not self.column:
            raise TypeError("column must be a non-empty string.")
        _validate_literal("sampling", self.sampling, {"uniform", "random", "center", "start"})
        _validate_literal("decoder", self.decoder, {"torchcodec", "decord", "cv2"})
        _validate_literal("output", self.output, {"torch", "numpy"})
        _validate_literal("layout", self.layout, {"TCHW", "CTHW", "THWC"})
        _validate_literal("dtype", self.dtype, {"float32", "uint8"})
        _validate_literal("scale", self.scale, {"zero_one", "none"})
        _validate_literal("crop", self.crop, {"none", "center", "random"})
        _validate_literal("pad", self.pad, {"error", "repeat_last", "loop"})
        if self.scale == "zero_one" and self.dtype != "float32":
            raise ValueError("scale='zero_one' requires dtype='float32'.")
        if self.resize is not None:
            if isinstance(self.resize, int):
                if self.resize < 1:
                    raise ValueError("resize must be >= 1 when provided as an int.")
            elif (
                not isinstance(self.resize, tuple)
                or len(self.resize) != 2
                or any(not isinstance(v, int) or v < 1 for v in self.resize)
            ):
                raise ValueError("resize must be an int or a tuple of two positive ints.")


def _validate_literal(name: str, value: str, allowed: set[str]) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}, got {value!r}.")


class Features(OrderedDict):
    """Ordered mapping of ``field_name -> FeatureType``.

    Generates a PyArrow schema via ``.to_arrow_schema()``.
    """

    def to_arrow_schema(self) -> pa.schema:
        fields = []
        for name, feat in self.items():
            if not isinstance(feat, FeatureType):
                raise TypeError(f"Feature '{name}' must be a FeatureType, got {type(feat).__name__}")
            metadata = feat.arrow_metadata()
            fields.append(pa.field(name, feat.to_arrow_type(), metadata=metadata or None))
        return pa.schema(fields)

    def fingerprint_data(self) -> str:
        # Preserve the historical dict-style payload so cache keys stay stable.
        return repr(dict(self))


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
