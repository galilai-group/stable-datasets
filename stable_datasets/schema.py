"""Feature and metadata schema definitions.

Each feature type maps itself to a PyArrow type for Arrow IPC serialization.
"""

from __future__ import annotations

import hashlib
import io
import mimetypes
import os
import shutil
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, NewType

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
    """Typed provenance + download metadata for one dataset builder."""

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
                    f"DatasetSource.assets['{key}'] must be a URL string or DownloadInfo, "
                    f"got {type(value).__name__}."
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


class Image(FeatureType):
    """Image feature. Stored as raw bytes in Arrow.

    *encode_format* controls how numpy arrays are encoded at cache-write
    time.  ``"PNG"`` (default) is lossless; ``"JPEG"`` is much faster and
    smaller for photographic RGB content.  The format is a cache-time
    concern — readers auto-detect from the bytes header.

    Uses ``large_binary`` (i64 offsets) rather than ``binary`` (i32) so
    that the column's cumulative bytes can exceed 2GB without overflowing
    PyArrow's compute kernels — an issue for any image dataset of
    ImageNet-1k scale.
    """

    def __init__(self, encode_format: str = "PNG"):
        self.encode_format = encode_format

    def to_arrow_type(self) -> pa.DataType:
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


@dataclass
class VideoRef:
    """Lazy reference to a cached video asset."""

    cell: Mapping[str, Any]
    cache_dir: Path | None = None
    _materialized_path: Path | None = field(default=None, init=False, repr=False)

    @property
    def mode(self) -> str:
        return self.cell["mode"]

    @property
    def extension(self) -> str:
        return self.cell["extension"]

    @property
    def media_type(self) -> str:
        return self.cell["media_type"]

    @property
    def size(self) -> int:
        return int(self.cell["size"])

    @property
    def checksum(self) -> str | None:
        return self.cell.get("checksum")

    @property
    def path(self) -> Path | None:
        if self.mode == "path":
            rel = self.cell.get("path")
            if rel is None:
                return None
            rel_path = Path(rel)
            if rel_path.is_absolute():
                return rel_path
            if self.cache_dir is None:
                return rel_path
            return self.cache_dir / "_assets" / "video" / rel_path

        if self.mode == "bytes":
            if self._materialized_path is None:
                suffix = self.extension or ".video"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as fh:
                    fh.write(self.bytes)
                    self._materialized_path = Path(fh.name)
            return self._materialized_path
        return None

    @property
    def bytes(self) -> bytes:
        data = self.cell.get("bytes")
        if data is not None:
            return data
        path = self.path
        if path is None:
            raise ValueError("VideoRef has neither bytes nor a path.")
        return path.read_bytes()

    def open(self, decoder: str = "torchcodec"):
        if decoder == "torchcodec":
            try:
                from torchcodec.decoders import VideoDecoder
            except ImportError as exc:
                raise ImportError(
                    "Video decoding with torchcodec requires installing "
                    "stable-datasets[video] or torchcodec."
                ) from exc
            return VideoDecoder(str(self.path))
        if decoder == "cv2":
            try:
                import cv2
            except ImportError as exc:
                raise ImportError("Video decoding with cv2 requires opencv-python.") from exc
            return cv2.VideoCapture(str(self.path))
        if decoder == "decord":
            try:
                from decord import VideoReader
            except ImportError as exc:
                raise ImportError("Video decoding with decord requires installing decord.") from exc
            return VideoReader(str(self.path))
        raise ValueError(f"Unknown video decoder {decoder!r}.")

    def get_frame_at(self, time: float, *, decoder: str = "torchcodec"):
        if decoder == "torchcodec":
            frame = self.open(decoder=decoder).get_frame_at(float(time))
            data = getattr(frame, "data", frame)
            try:
                import torch

                if isinstance(data, torch.Tensor):
                    data = data.detach().cpu().numpy()
            except ImportError:
                pass
            return data

        if decoder == "cv2":
            cap = self.open(decoder=decoder)
            try:
                import cv2

                cap.set(cv2.CAP_PROP_POS_MSEC, float(time) * 1000.0)
                ok, bgr = cap.read()
                if not ok:
                    raise ValueError(f"Could not decode frame at time {time}.")
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            finally:
                cap.release()

        raise NotImplementedError(f"get_frame_at is not implemented for decoder {decoder!r}.")


class Video(FeatureType):
    """Video feature with validated path, bytes, or specialized frame storage."""

    _DEFAULT_EXTENSIONS = (".mp4", ".avi", ".mov", ".webm", ".mkv")

    def __init__(
        self,
        storage: str = "path",
        allowed_extensions: tuple[str, ...] = _DEFAULT_EXTENSIONS,
    ):
        if storage not in ("path", "bytes", "frames"):
            raise ValueError("Video.storage must be one of 'path', 'bytes', or 'frames'.")
        self.storage = storage
        self.allowed_extensions = tuple(ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in allowed_extensions)

    def to_arrow_type(self) -> pa.DataType:
        return pa.struct(
            [
                pa.field("mode", pa.string()),
                pa.field("path", pa.string()),
                pa.field("bytes", pa.large_binary()),
                pa.field("extension", pa.string()),
                pa.field("media_type", pa.string()),
                pa.field("size", pa.int64()),
                pa.field("checksum", pa.string()),
            ]
        )

    def arrow_metadata(self) -> dict[bytes, bytes]:
        return {
            **super().arrow_metadata(),
            b"stable_datasets.video.storage": self.storage.encode(),
        }

    def encode(self, value, *, cache_dir: Path | None = None):
        if value is None:
            return None
        if self.storage == "frames":
            raise NotImplementedError(
                "Video(storage='frames') uses the specialized Lance video-frames layout."
            )
        if self.storage == "path":
            path, checksum = self._coerce_path_value(value)
            return self._encode_path(path, checksum=checksum, cache_dir=cache_dir)
        return self._encode_bytes(value)

    def format(
        self,
        value,
        *,
        format_type: str,
        decode_images: bool = True,
        cache_dir: Path | None = None,
    ):
        if value is None:
            return None
        if self.storage == "frames":
            if not hasattr(value, "shape") and isinstance(value, list):
                import numpy as np

                value = np.asarray(value, dtype=np.uint8)
            if format_type == "torch" and hasattr(value, "shape"):
                import torch

                return value if isinstance(value, torch.Tensor) else torch.from_numpy(value)
            return value
        if format_type == "raw":
            return value
        return VideoRef(value, cache_dir=cache_dir)

    def _coerce_path_value(self, value) -> tuple[Path, str | None]:
        if isinstance(value, Mapping):
            raw_path = value.get("path")
            checksum = value.get("checksum")
        else:
            raw_path = value
            checksum = None
        if not isinstance(raw_path, str | Path):
            raise TypeError(
                "Video(storage='path') values must be str, pathlib.Path, "
                "or {'path': ..., 'checksum': ...} mappings."
            )
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video path does not exist or is not a file: {path}")
        self._validate_extension(path.suffix)
        return path, checksum

    def _encode_path(self, path: Path, *, checksum: str | None, cache_dir: Path | None):
        if cache_dir is None:
            raise ValueError("Video(storage='path') requires a cache_dir for asset staging.")
        ext = path.suffix.lower()
        digest = _sha256_file(path)
        checksum = checksum or digest
        asset_dir = Path(cache_dir) / "_assets" / "video"
        asset_dir.mkdir(parents=True, exist_ok=True)
        staged = asset_dir / f"{digest}{ext}"
        if not staged.exists():
            try:
                os.link(path, staged)
            except OSError:
                shutil.copy2(path, staged)
        size = staged.stat().st_size
        return {
            "mode": "path",
            "path": staged.name,
            "bytes": None,
            "extension": ext,
            "media_type": _media_type_for_extension(ext),
            "size": int(size),
            "checksum": checksum,
        }

    def _encode_bytes(self, value):
        checksum = None
        ext = None
        if isinstance(value, Mapping):
            checksum = value.get("checksum")
            ext = value.get("extension")
            if value.get("bytes") is not None:
                value = value["bytes"]
            elif value.get("path") is not None:
                value = value["path"]
            else:
                raise TypeError("Video bytes mapping must contain 'bytes' or 'path'.")

        if isinstance(value, str | Path):
            path = Path(value)
            if not path.is_file():
                raise FileNotFoundError(f"Video path does not exist or is not a file: {path}")
            ext = ext or path.suffix
            self._validate_extension(ext)
            data = path.read_bytes()
        elif isinstance(value, bytes | bytearray | memoryview):
            data = bytes(value)
            ext = ext or self.allowed_extensions[0]
            self._validate_extension(ext)
        else:
            raise TypeError("Video(storage='bytes') values must be path-like or bytes-like.")

        ext = ext.lower() if str(ext).startswith(".") else f".{str(ext).lower()}"
        checksum = checksum or hashlib.sha256(data).hexdigest()
        return {
            "mode": "bytes",
            "path": None,
            "bytes": data,
            "extension": ext,
            "media_type": _media_type_for_extension(ext),
            "size": len(data),
            "checksum": checksum,
        }

    def _validate_extension(self, ext: str):
        ext = ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        if ext not in self.allowed_extensions:
            raise ValueError(
                f"Unsupported video extension {ext!r}. "
                f"Allowed: {list(self.allowed_extensions)}"
            )

    def fingerprint_data(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        if self.storage == "path" and self.allowed_extensions == self._DEFAULT_EXTENSIONS:
            return "Video(storage='path')"
        return f"Video(storage={self.storage!r}, allowed_extensions={self.allowed_extensions!r})"


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


class Array3D(FeatureType):
    """Fixed-shape 3D array (e.g. 3D medical volumes). Stored as flat bytes.

    Uses ``large_binary`` for the same reason as :class:`Image` — large
    volumes cumulatively exceed the i32 offset limit.
    """

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


class Features(dict):
    """Ordered dict of ``field_name -> FeatureType``.

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
        # Preserve the historical repr-based fingerprint payload so existing
        # non-video caches are not invalidated by the codec refactor.
        return repr(self)


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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _media_type_for_extension(ext: str) -> str:
    guessed, _ = mimetypes.guess_type(f"file{ext}")
    return guessed or "application/octet-stream"
