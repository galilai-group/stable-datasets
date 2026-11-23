"""Cache keys, split manifests, cache metadata, and opening a cache by the layout its metadata records.

The layouts themselves (how a split is written and read) are in stable_datasets.storage.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from filelock import FileLock

from . import _uri
from .schema import Features, FeatureType


def encode_example(example: dict, features: Features, *, cache_dir: Path | None = None) -> dict:
    """Encode a single example dict into Arrow-compatible values."""
    encoded = {}
    for key, value in example.items():
        feat = features.get(key)
        if isinstance(feat, FeatureType):
            encoded[key] = feat.encode(value, cache_dir=cache_dir)
        else:
            if hasattr(value, "item"):
                encoded[key] = value.item()
            else:
                encoded[key] = value
    return encoded


def _features_fingerprint(features: Features) -> str:
    """SHA-256 fingerprint of a Features dict for cache invalidation."""
    return hashlib.sha256(features.fingerprint_data().encode()).hexdigest()[:16]


def cache_fingerprint(
    cls_name: str,
    version: str,
    config_name: str,
    split: str,
    *,
    features: Features,
) -> str:
    """Deterministic cache directory name for a dataset variant + split.

    The features fingerprint is in the hash, so a schema change (including a
    column's storage, such as how a video is stored) gives a new cache.
    """
    # "lance" stays in the key: caches built when Arrow was the alternative keep their names.
    key = f"{cls_name}:{version}:{config_name}:{split}:lance:{_features_fingerprint(features)}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return f"{cls_name.lower()}_{config_name}_{split}_{digest}"


def split_manifest_name(cls_name: str, version: str, config_name: str) -> str:
    """File name of the split manifest of a dataset variant.

    The manifest lists the split names of the variant. A builder reads it
    before any download to find the cached splits.
    """
    key = f"{cls_name}:{version}:{config_name}:lance"
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return f"{cls_name.lower()}_{config_name}_{digest}.splits.json"


def read_split_manifest(uri: str | Path, storage_options: dict | None = None) -> list[str] | None:
    """Return the split names in a split manifest, or ``None`` if it does not exist."""
    uri = os.fspath(uri)
    if not _uri.exists(uri, storage_options):
        return None
    return json.loads(_uri.read_text(uri, storage_options))["splits"]


def write_split_manifest(uri: str | Path, split_names: list[str], storage_options: dict | None = None) -> None:
    """Write a split manifest."""
    _uri.write_text(os.fspath(uri), json.dumps({"splits": list(split_names)}, indent=2), storage_options)


_CACHE_FORMAT_VERSION = 1

_METADATA_FILE = "_metadata.json"

# Target size of one Lance fragment. More fragments let
# DataLoader workers read disjoint files.
DEFAULT_SHARD_SIZE_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_ROWS_PER_FILE = 1024 * 1024


def _encode_gen(generator, features, batch_size, num_workers, *, cache_dir: Path | None = None):
    """Wrap a generator with optional parallel encoding.

    When *num_workers* <= 0, encodes serially (zero overhead).
    When > 0, collects chunks of *batch_size* examples and encodes
    them in parallel using a thread pool (PIL operations release the GIL).
    """
    if num_workers <= 0:
        for key, example in generator:
            yield key, encode_example(example, features, cache_dir=cache_dir)
        return

    def encode_fn(ex):
        return encode_example(ex, features, cache_dir=cache_dir)

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        chunk = []
        for key, example in generator:
            chunk.append((key, example))
            if len(chunk) >= batch_size:
                encoded = list(pool.map(encode_fn, [ex for _, ex in chunk]))
                for (k, _), enc in zip(chunk, encoded):
                    yield k, enc
                chunk = []
        if chunk:
            encoded = list(pool.map(encode_fn, [ex for _, ex in chunk]))
            for (k, _), enc in zip(chunk, encoded):
                yield k, enc


def _writer_lock(cache_dir: str | os.PathLike):
    """Return a file lock next to a local cache, or a no-op for a remote URI."""
    path = _uri.local_path(cache_dir)
    if path is None:
        return contextlib.nullcontext()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Append to the full name. with_suffix() cuts a config name with a dot.
    return FileLock(str(path.parent / f"{path.name}.lock"))


class LanceCacheMeta:
    """Lightweight descriptor for a Lance-format cache on disk."""

    __slots__ = ("cache_dir", "num_rows", "schema_fingerprint", "shard_row_counts", "version")

    def __init__(
        self,
        cache_dir: str | Path,
        num_rows: int,
        schema_fingerprint: str,
        shard_row_counts: list[int] | None = None,
        version: int | None = None,
    ):
        self.cache_dir = os.fspath(cache_dir)
        self.num_rows = num_rows
        self.schema_fingerprint = schema_fingerprint
        self.shard_row_counts = shard_row_counts
        self.version = version


def _read_metadata(cache_dir: str | Path, storage_options: dict | None = None) -> dict:
    """Read ``_metadata.json`` from a local or remote cache directory."""
    meta_path = _uri.join(cache_dir, _METADATA_FILE)
    if not _uri.exists(meta_path, storage_options):
        raise FileNotFoundError(f"No metadata file at {meta_path}")
    return json.loads(_uri.read_text(meta_path, storage_options))


def _metadata_field(raw: dict, field: str, cache_dir: str | Path):
    if field not in raw:
        raise ValueError(
            f"Cache metadata at {_uri.join(cache_dir, _METADATA_FILE)} is missing required {field!r}. "
            "Rebuild the cache with the current stable-datasets version."
        )
    return raw[field]


def detect_cache_layout(cache_dir: str | Path, storage_options: dict | None = None) -> str:
    """Return the physical cache layout recorded in cache metadata."""
    return _metadata_field(_read_metadata(cache_dir, storage_options), "layout", cache_dir)


class CacheOpenResult:
    """Result of opening cache metadata into a backend."""

    __slots__ = ("backend", "num_rows", "layout", "metadata", "storage")

    def __init__(self, *, backend, num_rows: int, layout: str, metadata, storage=None):
        self.backend = backend
        self.num_rows = int(num_rows)
        self.layout = layout
        self.metadata = metadata
        self.storage = storage


def open_cache(
    cache_dir: str | Path,
    features: Features,
    *,
    backend_kwargs: dict | None = None,
    storage_options: dict | None = None,
) -> CacheOpenResult:
    """Open a cache directory and return the backend selected by its layout.

    ``cache_dir`` is a local path or a URI. ``backend_kwargs`` go to the backend and
    replace the values from the metadata. ``storage_options`` go to Lance
    and to the metadata reads.
    """
    from .storage.base import layout_class

    cache_dir = os.fspath(cache_dir)
    metadata = _read_metadata(cache_dir, storage_options)
    layout = _metadata_field(metadata, "layout", cache_dir)
    try:
        cls = layout_class(layout)
    except KeyError:
        raise ValueError(f"Unknown cache layout {layout!r} in {cache_dir}") from None
    opened = cls.open(
        cache_dir, features, metadata, backend_kwargs=dict(backend_kwargs or {}), storage_options=storage_options
    )
    if opened.storage is None:
        declared = cls.from_features(features)
        opened.storage = declared if isinstance(declared, cls) else cls()
    return opened


def read_lance_cache_meta(cache_dir: str | Path, storage_options: dict | None = None) -> LanceCacheMeta:
    """Read metadata from a Lance-format cache directory.

    Returns a :class:`LanceCacheMeta` with the cached row count and
    schema fingerprint populated from ``_metadata.json``. Deliberately
    does NOT open the underlying Lance dataset: that would initialize
    Lance's tokio runtime in the caller's process, which is a
    DataLoader-fork footgun. Row count comes from the metadata file,
    not from ``ds.count_rows()``.
    """
    return lance_cache_meta(_read_metadata(cache_dir, storage_options), cache_dir)


def lance_cache_meta(raw: dict, cache_dir: str | Path) -> LanceCacheMeta:
    """A :class:`LanceCacheMeta` from the ``_metadata.json`` of a Lance cache, after its format checks."""
    if raw.get("format") != "lance":
        raise ValueError(f"Not a Lance cache: {cache_dir} (format={raw.get('format')!r})")
    _check_format_version(raw)
    return LanceCacheMeta(
        cache_dir=cache_dir,
        num_rows=raw["num_rows"],
        schema_fingerprint=raw.get("schema_fingerprint", ""),
        shard_row_counts=raw.get("shard_row_counts"),
        version=raw["version"],
    )


def _check_format_version(raw: dict) -> None:
    """Raise ``ValueError`` if cache metadata has a different format version."""
    fmt_version = raw.get("cache_format_version")
    if fmt_version != _CACHE_FORMAT_VERSION:
        raise ValueError(f"Unsupported cache format version {fmt_version} (expected {_CACHE_FORMAT_VERSION})")


def _check_schema_fingerprint(found: str, features: Features, cache_dir: Path) -> None:
    """Raise ``ValueError`` if a cache was written with different features."""
    expected_fp = _features_fingerprint(features)
    if found != expected_fp:
        raise ValueError(
            f"Schema fingerprint mismatch in {cache_dir}: cache has {found!r}, "
            f"expected {expected_fp!r}.  Delete the cache and rebuild."
        )
