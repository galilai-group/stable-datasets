"""Generator-to-Arrow caching pipeline.

Replaces HuggingFace's ``download_and_prepare()`` / ``ArrowWriter`` with direct
PyArrow IPC (Feather v2) writing.  Supports batched writes to limit memory and
file-locking for concurrent access safety.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from filelock import FileLock
from loguru import logger as logging
from PIL import Image as PILImage

from .schema import Array3D, Features, Image, Sequence, Value, Video


# ---------------------------------------------------------------------------
# Example encoding
# ---------------------------------------------------------------------------


def _encode_image(img) -> bytes | None:
    """Encode a PIL Image, numpy array, or file path to PNG bytes."""
    if img is None:
        return None
    if isinstance(img, PILImage.Image):
        buf = io.BytesIO()
        fmt = "PNG" if img.mode in ("RGBA", "LA", "PA", "P") else "PNG"
        img.save(buf, format=fmt)
        return buf.getvalue()
    if isinstance(img, np.ndarray):
        pil_img = PILImage.fromarray(img)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()
    if isinstance(img, (str, Path)):
        with open(img, "rb") as f:
            return f.read()
    if isinstance(img, bytes):
        return img
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
        elif isinstance(feat, Video):
            encoded[key] = str(value) if value is not None else None
        elif isinstance(feat, Sequence):
            # Convert to plain Python list for Arrow
            if hasattr(value, "tolist"):
                encoded[key] = value.tolist()
            else:
                encoded[key] = list(value) if value is not None else None
        else:
            # Scalars: int, float, str, bool — convert numpy scalars to Python
            if hasattr(value, "item"):
                encoded[key] = value.item()
            else:
                encoded[key] = value
    return encoded


# ---------------------------------------------------------------------------
# Cache fingerprint
# ---------------------------------------------------------------------------


def cache_fingerprint(cls_name: str, version: str, config_name: str, split: str) -> str:
    """Deterministic cache filename for a dataset variant + split."""
    key = f"{cls_name}:{version}:{config_name}:{split}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return f"{cls_name.lower()}_{config_name}_{split}_{digest}.arrow"


# ---------------------------------------------------------------------------
# Arrow IPC write / read
# ---------------------------------------------------------------------------


def write_arrow_cache(
    generator,
    features: Features,
    cache_path: Path,
    batch_size: int = 1000,
) -> pa.Table:
    """Consume a ``_generate_examples()`` generator and write to Arrow IPC file.

    Writes in batches of ``batch_size`` rows to limit peak memory.
    Returns the final table for immediate use.
    """
    schema = features.to_arrow_schema()
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_path.with_suffix(".arrow.lock")

    all_batches: list[pa.RecordBatch] = []
    batch_rows: dict[str, list] = {name: [] for name in features}

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
        # Reset
        for col_name in batch_rows:
            batch_rows[col_name] = []
        return batch

    count = 0
    for _key, example in generator:
        encoded = encode_example(example, features)
        for col_name in features:
            batch_rows[col_name].append(encoded.get(col_name))
        count += 1

        if count % batch_size == 0:
            batch = _flush_batch()
            if batch is not None:
                all_batches.append(batch)

    # Flush remaining
    batch = _flush_batch()
    if batch is not None:
        all_batches.append(batch)

    if not all_batches:
        # Empty dataset — create a zero-row table
        table = pa.table({name: pa.array([], type=features[name].to_arrow_type()) for name in features}, schema=schema)
    else:
        table = pa.Table.from_batches(all_batches, schema=schema)

    # Write to disk with file lock
    with FileLock(str(lock_path)):
        with pa.OSFile(str(cache_path), "wb") as sink:
            writer = ipc.new_file(sink, schema)
            for batch in all_batches:
                writer.write_batch(batch)
            writer.close()

    logging.info(f"Cached {count} examples to {cache_path}")
    return table


def read_arrow_cache(cache_path: Path) -> pa.Table:
    """Read a cached Arrow IPC file. Memory-maps for zero-copy reads."""
    cache_path = Path(cache_path)
    mmap = pa.memory_map(str(cache_path), "r")
    reader = ipc.open_file(mmap)
    return reader.read_all()
