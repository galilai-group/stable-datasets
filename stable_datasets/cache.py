"""Generator-to-Arrow sharded caching pipeline.

Writes dataset examples to a directory of PyArrow IPC (Feather v2) shard
files.  Peak memory during writes is bounded to ~1 batch, and the sharded
layout supports efficient sequential reads for training workloads.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from filelock import FileLock
from loguru import logger as logging
from PIL import Image as PILImage

from .schema import Array3D, ClassLabel, Features, Image, Sequence, Video


# Encoding helpers


def _encode_image(img) -> bytes | None:
    """Encode an image to bytes, preserving the original format when possible.

    - Raw ``bytes`` pass through unchanged.
    - File paths are read as-is (JPEG stays JPEG, PNG stays PNG).
    - PIL Images opened from a file retain their ``.format``; we re-encode in
      the same format.  Programmatically-created images default to PNG.
    - NumPy arrays are converted via PIL and saved as PNG.
    """
    if img is None:
        return None
    if isinstance(img, bytes):
        return img
    if isinstance(img, str | Path):
        with open(img, "rb") as f:
            return f.read()
    if isinstance(img, PILImage.Image):
        buf = io.BytesIO()
        # Preserve source format when available (e.g. JPEG from Image.open).
        # Fall back to PNG for images with alpha or no known source format.
        fmt = getattr(img, "format", None)
        if fmt is None or img.mode in ("RGBA", "LA", "PA", "P"):
            fmt = "PNG"
        img.save(buf, format=fmt)
        return buf.getvalue()
    if isinstance(img, np.ndarray):
        pil_img = PILImage.fromarray(img)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()
    raise TypeError(f"Cannot encode image of type {type(img)}")


def _encode_array3d(arr, feat: Array3D) -> bytes | None:
    """Encode a numpy array to flat bytes for Arrow storage."""
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=feat.dtype)
    return arr.tobytes()


def encode_example(example: dict, features: Features) -> dict:
    """Encode a single example dict into Arrow-compatible values."""
    encoded = {}
    for key, value in example.items():
        feat = features.get(key)
        if isinstance(feat, Image):
            encoded[key] = _encode_image(value)
        elif isinstance(feat, Array3D):
            encoded[key] = _encode_array3d(value, feat)
        elif isinstance(feat, ClassLabel):
            if isinstance(value, str):
                encoded[key] = feat.str2int(value)
            else:
                encoded[key] = value
        elif isinstance(feat, Video):
            encoded[key] = str(value) if value is not None else None
        elif isinstance(feat, Sequence):
            if hasattr(value, "tolist"):
                encoded[key] = value.tolist()
            else:
                encoded[key] = list(value) if value is not None else None
        else:
            if hasattr(value, "item"):
                encoded[key] = value.item()
            else:
                encoded[key] = value
    return encoded


def _features_fingerprint(features: Features) -> str:
    """SHA-256 fingerprint of a Features dict for cache invalidation."""
    return hashlib.sha256(repr(features).encode()).hexdigest()[:16]


def cache_fingerprint(cls_name: str, version: str, config_name: str, split: str) -> str:
    """Deterministic cache directory name for a dataset variant + split."""
    key = f"{cls_name}:{version}:{config_name}:{split}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return f"{cls_name.lower()}_{config_name}_{split}_{digest}"


# Sharded readers/writers

_CACHE_FORMAT_VERSION = 1

_SHARD_NAME_FMT = "shard-{:05d}.arrow"
_METADATA_FILE = "_metadata.json"

# Default shard target: 256 MiB
DEFAULT_SHARD_SIZE_BYTES = 256 * 1024 * 1024


def write_sharded_arrow_cache(
    generator,
    features: Features,
    cache_dir: Path,
    *,
    shard_size_bytes: int = DEFAULT_SHARD_SIZE_BYTES,
    batch_size: int = 1000,
) -> ShardedCacheMeta:
    """Consume a generator and write to a directory of Arrow IPC shards.

    Batches are flushed every *batch_size* rows.  After each flush the
    cumulative ``RecordBatch.nbytes`` for the current shard is checked;
    when it exceeds *shard_size_bytes* the shard is closed.  The next
    shard is opened lazily when the next batch is ready, so there are
    never trailing empty shards.

    .. note::

       *shard_size_bytes* is an **approximate target** based on Arrow
       in-memory batch sizes, not exact on-disk file sizes.  Actual shard
       files may be somewhat larger or smaller due to IPC framing, batch
       granularity, and compression differences.

    An empty generator produces zero shards (``num_shards == 0``).

    The completed cache directory contains:

    * ``shard-NNNNN.arrow`` — zero or more IPC files
    * ``_metadata.json`` — row counts, shard list, format version,
      schema fingerprint

    Writing is atomic: shards are first written to a temporary directory
    and renamed into place on success.

    Returns a :class:`ShardedCacheMeta` describing the cache.
    """
    cache_dir = Path(cache_dir)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir.with_suffix(".lock")
    schema = features.to_arrow_schema()

    # Work in a temp dir next to the final location; rename on success.
    tmp_dir = Path(tempfile.mkdtemp(dir=cache_dir.parent, prefix=f".{cache_dir.name}_tmp_"))

    batch_rows: dict[str, list] = {name: [] for name in features}

    # Per-shard bookkeeping
    shard_filenames: list[str] = []
    shard_row_counts: list[int] = []
    shard_idx = 0
    shard_bytes = 0  # cumulative RecordBatch.nbytes for current shard
    shard_rows = 0
    writer: ipc.RecordBatchFileWriter | None = None
    sink: pa.OSFile | None = None

    def _ensure_shard_open():
        """Open a new shard if one is not already open."""
        nonlocal writer, sink, shard_bytes, shard_rows
        if writer is not None:
            return
        fname = _SHARD_NAME_FMT.format(shard_idx)
        shard_filenames.append(fname)
        sink = pa.OSFile(str(tmp_dir / fname), "wb")
        writer = ipc.new_file(sink, schema)
        shard_bytes = 0
        shard_rows = 0

    def _close_shard():
        nonlocal writer, sink, shard_idx
        if writer is None:
            return
        writer.close()
        writer = None
        sink.close()
        sink = None
        shard_row_counts.append(shard_rows)
        shard_idx += 1

    def _flush_batch() -> pa.RecordBatch | None:
        if not batch_rows[next(iter(batch_rows))]:
            return None
        arrays = []
        for col_name in features:
            feat = features[col_name]
            col_data = batch_rows[col_name]
            arr = pa.array(col_data, type=feat.to_arrow_type())
            arrays.append(arr)
        batch = pa.record_batch(arrays, schema=schema)
        for col_name in batch_rows:
            batch_rows[col_name] = []
        return batch

    def _write_batch(batch: pa.RecordBatch):
        nonlocal shard_bytes, shard_rows
        _ensure_shard_open()
        writer.write_batch(batch)
        shard_bytes += batch.nbytes
        shard_rows += batch.num_rows

    total_count = 0

    try:
        with FileLock(str(lock_path)):
            for _key, example in generator:
                encoded = encode_example(example, features)
                for col_name in features:
                    batch_rows[col_name].append(encoded.get(col_name))
                total_count += 1

                if total_count % batch_size == 0:
                    batch = _flush_batch()
                    if batch is not None:
                        _write_batch(batch)
                        # Rotate shard if over budget
                        if shard_bytes >= shard_size_bytes:
                            _close_shard()

            # Flush remaining rows
            batch = _flush_batch()
            if batch is not None:
                _write_batch(batch)

            _close_shard()

            # Write metadata
            meta = {
                "cache_format_version": _CACHE_FORMAT_VERSION,
                "schema_fingerprint": _features_fingerprint(features),
                "num_rows": total_count,
                "num_shards": len(shard_filenames),
                "shard_row_counts": shard_row_counts,
                "shard_filenames": shard_filenames,
                "shard_size_target_bytes": shard_size_bytes,
            }
            (tmp_dir / _METADATA_FILE).write_text(json.dumps(meta, indent=2))

            # Atomic publish: rename temp dir → final cache dir
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            os.rename(str(tmp_dir), str(cache_dir))

    except BaseException:
        # Clean up temp dir on failure
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    logging.info(f"Cached {total_count} examples in {len(shard_filenames)} shard(s) to {cache_dir}")

    return ShardedCacheMeta(
        cache_dir=cache_dir,
        num_rows=total_count,
        num_shards=len(shard_filenames),
        shard_filenames=shard_filenames,
        shard_row_counts=shard_row_counts,
        schema_fingerprint=meta["schema_fingerprint"],
    )


class ShardedCacheMeta:
    """Lightweight descriptor for a sharded Arrow cache on disk."""

    __slots__ = (
        "cache_dir",
        "num_rows",
        "num_shards",
        "shard_filenames",
        "shard_row_counts",
        "schema_fingerprint",
    )

    def __init__(
        self,
        cache_dir: Path,
        num_rows: int,
        num_shards: int,
        shard_filenames: list[str],
        shard_row_counts: list[int],
        schema_fingerprint: str,
    ):
        self.cache_dir = Path(cache_dir)
        self.num_rows = num_rows
        self.num_shards = num_shards
        self.shard_filenames = shard_filenames
        self.shard_row_counts = shard_row_counts
        self.schema_fingerprint = schema_fingerprint

    @property
    def shard_paths(self) -> list[Path]:
        return [self.cache_dir / f for f in self.shard_filenames]


def read_sharded_cache_meta(cache_dir: Path) -> ShardedCacheMeta:
    """Read metadata from a sharded cache directory.

    Validates that all shard files and metadata exist and are internally
    consistent.  Raises ``ValueError`` on corruption.
    """
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / _METADATA_FILE

    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata file at {meta_path}")

    raw = json.loads(meta_path.read_text())

    # Version check
    fmt_version = raw.get("cache_format_version")
    if fmt_version != _CACHE_FORMAT_VERSION:
        raise ValueError(f"Unsupported cache format version {fmt_version} (expected {_CACHE_FORMAT_VERSION})")

    shard_filenames = raw["shard_filenames"]
    shard_row_counts = raw["shard_row_counts"]
    num_rows = raw["num_rows"]
    num_shards = raw["num_shards"]

    # Consistency checks
    if len(shard_filenames) != num_shards:
        raise ValueError(f"Metadata claims {num_shards} shards but lists {len(shard_filenames)} filenames")
    if len(shard_row_counts) != num_shards:
        raise ValueError(f"Metadata claims {num_shards} shards but has {len(shard_row_counts)} row counts")
    if sum(shard_row_counts) != num_rows:
        raise ValueError(f"Sum of shard_row_counts ({sum(shard_row_counts)}) != num_rows ({num_rows})")

    # Check shard files exist
    for fname in shard_filenames:
        if not (cache_dir / fname).exists():
            raise ValueError(f"Missing shard file: {cache_dir / fname}")

    return ShardedCacheMeta(
        cache_dir=cache_dir,
        num_rows=num_rows,
        num_shards=num_shards,
        shard_filenames=shard_filenames,
        shard_row_counts=shard_row_counts,
        schema_fingerprint=raw.get("schema_fingerprint", ""),
    )


def validate_sharded_cache(cache_dir: Path, features: Features) -> ShardedCacheMeta:
    """Read and validate a sharded cache, checking the schema fingerprint.

    Raises ``ValueError`` if the cache is inconsistent or the schema has changed.
    """
    meta = read_sharded_cache_meta(cache_dir)
    expected_fp = _features_fingerprint(features)
    if meta.schema_fingerprint and meta.schema_fingerprint != expected_fp:
        raise ValueError(
            f"Schema fingerprint mismatch: cache has {meta.schema_fingerprint}, "
            f"expected {expected_fp}.  Delete the cache and rebuild."
        )
    return meta


def read_shard(shard_path: Path) -> pa.Table:
    """Memory-map a single shard file and return its table."""
    mmap = pa.memory_map(str(shard_path), "r")
    reader = ipc.open_file(mmap)
    return reader.read_all()
