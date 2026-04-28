"""Video feature codec and lazy reference objects."""

from __future__ import annotations

import hashlib
import mimetypes
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa

from .base import FeatureType


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
        self.allowed_extensions = tuple(
            ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in allowed_extensions
        )

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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _media_type_for_extension(ext: str) -> str:
    guessed, _ = mimetypes.guess_type(f"file{ext}")
    return guessed or "application/octet-stream"
