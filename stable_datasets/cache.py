"""Generator-to-Arrow sharded caching pipeline.

Writes dataset examples to a directory of PyArrow IPC (Feather v2) shard
files.  Peak memory during writes is bounded to ~1 batch, and the sharded
layout supports efficient sequential reads for training workloads.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
from filelock import FileLock
from loguru import logger as logging

from . import _uri
from .schema import Features, FeatureType, Video


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
    storage_format: str = "arrow",
    *,
    features: Features,
) -> str:
    """Deterministic cache directory name for a dataset variant + split.

    ``storage_format`` is always included in the hash so Arrow and Lance
    caches for the same dataset coexist at different paths. The features
    fingerprint is in the hash too, so a schema change gives a new cache.
    """
    key = f"{cls_name}:{version}:{config_name}:{split}:{storage_format}:{_features_fingerprint(features)}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return f"{cls_name.lower()}_{config_name}_{split}_{digest}"


def split_manifest_name(cls_name: str, version: str, config_name: str, storage_format: str) -> str:
    """File name of the split manifest of a dataset variant.

    The manifest lists the split names of the variant. A builder reads it
    before any download to find the cached splits.
    """
    key = f"{cls_name}:{version}:{config_name}:{storage_format}"
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


# Sharded readers/writers

_CACHE_FORMAT_VERSION = 1

_SHARD_NAME_FMT = "shard-{:05d}.arrow"
_METADATA_FILE = "_metadata.json"

# Target size of one Arrow shard or one Lance fragment. More shards let
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


def write_sharded_arrow_cache(
    generator,
    features: Features,
    cache_dir: Path,
    *,
    shard_size_bytes: int = DEFAULT_SHARD_SIZE_BYTES,
    batch_size: int = 1000,
    compression: str | None = None,
    num_encode_workers: int = 0,
    single_file: bool = False,
    lineage: dict | None = None,
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

    * ``data-XXXXXXXX/shard-NNNNN.arrow`` — zero or more IPC files
    * ``_metadata.json`` — row counts, shard list, format version,
      schema fingerprint

    Publishing is atomic for readers. The writer puts the shards in a new
    ``data-*`` directory, then replaces ``_metadata.json``, which lists
    them. The cache directory always exists, and a reader sees the old or
    the new shard list. The writer keeps the previous ``data-*``
    directory, so a reader that opened the previous cache can still read
    its shards. It removes older ``data-*`` directories.

    Parameters
    ----------
    compression : str or None
        IPC buffer compression codec (e.g. ``"zstd"``, ``"lz4"``).
        Decompression on read is automatic.
    num_encode_workers : int
        When > 0, encode examples in parallel using a thread pool.

    Returns a :class:`ShardedCacheMeta` describing the cache.
    """
    cache_dir = Path(cache_dir)
    created_cache_dir = not cache_dir.exists()
    schema = features.to_arrow_schema()

    if single_file:
        shard_size_bytes = float("inf")

    ipc_options = ipc.IpcWriteOptions(compression=compression) if compression else None

    data_dir: Path | None = None
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
        shard_filenames.append(f"{data_dir.name}/{fname}")
        sink = pa.OSFile(str(data_dir / fname), "wb")
        if ipc_options:
            writer = ipc.new_file(sink, schema, options=ipc_options)
        else:
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
        with _writer_lock(cache_dir):
            # Make the data directory under the lock, so another writer cannot remove it.
            cache_dir.mkdir(parents=True, exist_ok=True)
            data_dir = Path(tempfile.mkdtemp(dir=cache_dir, prefix="data-"))
            # Wrap generator with optional parallel encoding
            encoded_gen = _encode_gen(generator, features, batch_size, num_encode_workers, cache_dir=data_dir)
            for _key, encoded in encoded_gen:
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
                "format": "arrow",
                "layout": "arrow-shards",
                "schema_fingerprint": _features_fingerprint(features),
                "num_rows": total_count,
                "num_shards": len(shard_filenames),
                "shard_row_counts": shard_row_counts,
                "shard_filenames": shard_filenames,
                "shard_size_target_bytes": None if shard_size_bytes == float("inf") else shard_size_bytes,
            }
            if compression:
                meta["compression"] = compression
            if lineage:
                meta["lineage"] = lineage
            # Atomic publish: replace the metadata, then remove the data directories
            # that neither the new nor the previous metadata lists.
            meta_path = cache_dir / _METADATA_FILE
            keep = {data_dir.name}
            try:
                keep.update(name.split("/")[0] for name in json.loads(meta_path.read_text())["shard_filenames"])
            except (OSError, ValueError, KeyError):
                pass  # No readable previous cache: keep only the new data directory.
            _uri.write_text(str(meta_path), json.dumps(meta, indent=2))
            for old_dir in cache_dir.glob("data-*"):
                if old_dir.name not in keep:
                    shutil.rmtree(old_dir, ignore_errors=True)

    except BaseException:
        if data_dir is not None:
            shutil.rmtree(data_dir, ignore_errors=True)
        if created_cache_dir and cache_dir.is_dir() and not any(cache_dir.iterdir()):
            cache_dir.rmdir()
        raise

    logging.info(f"Cached {total_count} examples in {len(shard_filenames)} shard(s) to {cache_dir}")

    return ShardedCacheMeta(
        cache_dir=cache_dir,
        num_rows=total_count,
        num_shards=len(shard_filenames),
        shard_filenames=shard_filenames,
        shard_row_counts=shard_row_counts,
        schema_fingerprint=meta["schema_fingerprint"],
        compression=compression,
    )


def write_lance_cache(
    generator,
    features: Features,
    cache_dir: str | Path,
    *,
    batch_size: int = 1000,
    num_encode_workers: int = 0,
    max_rows_per_file: int = DEFAULT_MAX_ROWS_PER_FILE,
    max_bytes_per_file: int = DEFAULT_SHARD_SIZE_BYTES,
    lineage: dict | None = None,
    storage_options: dict | None = None,
) -> LanceCacheMeta:
    """Consume a generator and write directly to a Lance dataset.

    Mirrors :func:`write_sharded_arrow_cache` in shape (same generator
    contract, same features, same encode pipeline) but writes a Lance
    dataset via ``lance.write_dataset`` instead of Arrow IPC shards. No
    intermediate Arrow IPC file is produced -- the encoded rows stream
    into Lance via a :class:`pa.RecordBatchReader`, so the native Lance
    write path is used end-to-end.

    ``cache_dir`` is a local path or a URI such as ``s3://bucket/prefix``.
    A Lance commit publishes the dataset atomically, and the writer then
    writes ``_metadata.json``. A cache without ``_metadata.json`` is
    incomplete. The completed cache directory contains:

    * Lance dataset files (``_versions/``, ``data/``, manifest)
    * ``_metadata.json`` -- row count, row count of each fragment, format
      marker, schema fingerprint

    Parameters
    ----------
    batch_size : int
        Rows per ``pa.RecordBatch`` flushed to the Lance writer. Larger
        batches reduce per-call overhead; smaller batches reduce peak
        memory during writing.
    num_encode_workers : int
        When > 0, encode examples in parallel using a thread pool
        (same contract as the Arrow writer).
    max_rows_per_file : int
        Maximum number of rows in one Lance fragment.
    max_bytes_per_file : int
        Approximate target size of one Lance fragment. The writer
        converts it to a row limit from the size of the first batch.
    lineage : dict, optional
        Optional metadata blob written into ``_metadata.json``.
    storage_options : dict, optional
        Object store options for Lance and for the metadata file.
    """
    import lance

    cache_dir = os.fspath(cache_dir)
    schema = features.to_arrow_schema()

    # Counter state is read after Lance consumes the batch stream.
    counter = {"rows": 0}

    # Video(storage="path") stages assets in a local cache directory only.
    encoded_gen = _encode_gen(
        generator, features, batch_size, num_encode_workers, cache_dir=_uri.local_path(cache_dir)
    )

    def batch_stream():
        batch_rows: dict[str, list] = {name: [] for name in features}
        for _key, encoded in encoded_gen:
            for col_name in features:
                batch_rows[col_name].append(encoded.get(col_name))
            counter["rows"] += 1
            if counter["rows"] % batch_size == 0:
                arrays = [pa.array(batch_rows[name], type=features[name].to_arrow_type()) for name in features]
                yield pa.record_batch(arrays, schema=schema)
                for col in batch_rows:
                    batch_rows[col] = []
        if batch_rows[next(iter(batch_rows))]:
            arrays = [pa.array(batch_rows[name], type=features[name].to_arrow_type()) for name in features]
            yield pa.record_batch(arrays, schema=schema)

    with _writer_lock(cache_dir):
        batches = batch_stream()
        first = next(batches, None)
        rows_per_file = max_rows_per_file
        if first is not None:
            batches = itertools.chain([first], batches)
            if first.nbytes > 0:
                # Lance v2 files do not apply max_bytes_per_file. Derive a row limit instead.
                rows_per_file = min(rows_per_file, max(1, max_bytes_per_file * first.num_rows // first.nbytes))
        rbr = pa.RecordBatchReader.from_batches(schema, batches)
        dataset = lance.write_dataset(
            rbr,
            cache_dir,
            mode="overwrite",
            max_rows_per_file=rows_per_file,
            max_bytes_per_file=max_bytes_per_file,
            storage_options=storage_options,
        )

        meta = {
            "cache_format_version": _CACHE_FORMAT_VERSION,
            "format": "lance",
            "layout": "lance-rows",
            "schema_fingerprint": _features_fingerprint(features),
            "num_rows": counter["rows"],
            "shard_row_counts": [fragment.count_rows() for fragment in dataset.get_fragments()],
            "version": dataset.version,
        }
        if lineage:
            meta["lineage"] = lineage
        _uri.write_text(_uri.join(cache_dir, _METADATA_FILE), json.dumps(meta, indent=2), storage_options)

    logging.info(f"Cached {counter['rows']} examples in Lance format to {cache_dir}")

    return LanceCacheMeta(
        cache_dir=cache_dir,
        num_rows=counter["rows"],
        schema_fingerprint=meta["schema_fingerprint"],
        shard_row_counts=meta["shard_row_counts"],
        version=meta["version"],
    )


_VIDEO_FRAME_SCHEMA = pa.schema(
    [
        ("video_id", pa.int32()),
        ("frame_idx", pa.int32()),
        ("bytes", pa.large_binary()),
    ]
)


def _encode_one_video_frames(task):
    """Decode one video path into WebP-encoded frame blobs."""
    video_id, path_str, quality, resize = task
    try:
        import cv2

        cap = cv2.VideoCapture(path_str)
        if not cap.isOpened():
            return ("error", video_id, path_str, "cv2.VideoCapture failed to open")
        enc_params = [int(cv2.IMWRITE_WEBP_QUALITY), int(quality)]
        blobs: list[bytes] = []
        out_h = out_w = None
        do_resize = None

        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if do_resize is None:
                h0, w0 = bgr.shape[:2]
                do_resize = resize is not None and max(h0, w0) > int(resize)
                out_h = int(resize) if do_resize else h0
                out_w = int(resize) if do_resize else w0
            if do_resize:
                bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".webp", bgr, enc_params)
            if not ok:
                cap.release()
                return ("error", video_id, path_str, f"webp encode failed at t={len(blobs)}")
            blobs.append(buf.tobytes())
        cap.release()
        if not blobs:
            return ("error", video_id, path_str, "empty video")
        return ("ok", video_id, path_str, len(blobs), int(out_h), int(out_w), blobs)
    except Exception as exc:
        return ("error", video_id, path_str, f"{type(exc).__name__}: {exc}")


def _video_input_to_path(value, *, tmp_dir: Path, allowed_extensions: tuple[str, ...]) -> tuple[Path, str | None]:
    checksum = None
    extension = None
    if isinstance(value, dict):
        checksum = value.get("checksum")
        extension = value.get("extension")
        if value.get("path") is not None:
            value = value["path"]
        elif value.get("bytes") is not None:
            value = value["bytes"]
        else:
            raise TypeError("Video frame values must contain 'path' or 'bytes'.")

    if isinstance(value, str | Path):
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"Video path does not exist or is not a file: {path}")
        ext = path.suffix.lower()
        if ext not in allowed_extensions:
            raise ValueError(f"Unsupported video extension {ext!r}. Allowed: {list(allowed_extensions)}")
        return path, checksum

    if isinstance(value, bytes | bytearray | memoryview):
        ext = extension or allowed_extensions[0]
        ext = ext.lower() if str(ext).startswith(".") else f".{str(ext).lower()}"
        if ext not in allowed_extensions:
            raise ValueError(f"Unsupported video extension {ext!r}. Allowed: {list(allowed_extensions)}")
        digest = hashlib.sha256(bytes(value)).hexdigest()
        input_dir = tmp_dir / "_video_inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        path = input_dir / f"{digest}{ext}"
        if not path.exists():
            path.write_bytes(bytes(value))
        return path, checksum or digest

    raise TypeError("Video frame values must be path-like or bytes-like.")


def write_lance_video_frames_cache(
    generator,
    features: Features,
    cache_dir: str | Path,
    *,
    video_column: str | None = None,
    quality: int = 65,
    resize: int | None = None,
    workers: int | None = None,
    skip_corrupt: bool = True,
    lineage: dict | None = None,
    storage_options: dict | None = None,
) -> LanceCacheMeta:
    """Write a specialized Lance row-per-frame video cache.

    Each input example contributes one source video. The physical Lance
    dataset stores one WebP-encoded frame per row. Segment sampling is a
    read-time concern handled by :class:`LanceVideoFramesBackend`.

    ``cache_dir`` is a local path or a URI. As in :func:`write_lance_cache`,
    a Lance commit publishes the frames and the writer then writes
    ``_metadata.json``.
    """
    import lance

    frame_columns = [name for name, feat in features.items() if isinstance(feat, Video) and feat.storage == "frames"]
    if video_column is None:
        if len(frame_columns) != 1:
            raise ValueError(
                "write_lance_video_frames_cache requires exactly one Video(storage='frames') "
                "column or an explicit video_column."
            )
        video_column = frame_columns[0]
    feat = features[video_column]
    if not isinstance(feat, Video) or feat.storage != "frames":
        raise TypeError(f"{video_column!r} must be a Video(storage='frames') feature.")

    cache_dir = os.fspath(cache_dir)
    local_dir = _uri.local_path(cache_dir)
    if local_dir is not None:
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        input_tmp_dir = Path(tempfile.mkdtemp(dir=local_dir.parent, prefix=f".{local_dir.name}_inputs_"))
    else:
        input_tmp_dir = Path(tempfile.mkdtemp(prefix=".video_frames_inputs_"))
    tasks = []
    examples: list[dict] = []
    try:
        for video_id, (key, example) in enumerate(generator):
            path, checksum = _video_input_to_path(
                example[video_column],
                tmp_dir=input_tmp_dir,
                allowed_extensions=feat.allowed_extensions,
            )
            metadata = {
                k: v
                for k, v in example.items()
                if k != video_column and isinstance(v, str | int | float | bool | type(None))
            }
            examples.append(
                {
                    "key": key if isinstance(key, str | int | float | bool | type(None)) else str(key),
                    "path": str(path),
                    "checksum": checksum,
                    "metadata": metadata,
                }
            )
            tasks.append((video_id, str(path), int(quality), resize))

        if not tasks:
            raise ValueError("Cannot build a lance-video-frames cache from an empty generator.")

        n_workers = int(workers) if workers else max(1, os.cpu_count() or 1)
        video_records: list[dict] = []

        def batch_stream(results):
            row = 0
            for result in results:
                tag = result[0]
                if tag == "error":
                    _, video_id, path_str, msg = result
                    logging.warning(f"skipping {path_str}: {msg}")
                    if not skip_corrupt:
                        raise RuntimeError(f"failed on {path_str}: {msg}")
                    continue
                _, video_id, path_str, frames, height, width, blobs = result
                original = examples[int(video_id)]
                video_records.append(
                    {
                        "id": int(video_id),
                        "key": original["key"],
                        "path": path_str,
                        "checksum": original["checksum"],
                        "T": int(frames),
                        "H": int(height),
                        "W": int(width),
                        "start_row": int(row),
                        "metadata": original["metadata"],
                    }
                )
                yield pa.record_batch(
                    [
                        pa.array([video_id] * frames, type=pa.int32()),
                        pa.array(range(frames), type=pa.int32()),
                        pa.array(blobs, type=pa.large_binary()),
                    ],
                    schema=_VIDEO_FRAME_SCHEMA,
                )
                row += frames

        with _writer_lock(cache_dir), contextlib.ExitStack() as stack:
            # Write each video when it is encoded. Do not keep all encoded videos in memory.
            if n_workers <= 1:
                results = map(_encode_one_video_frames, tasks)
            else:
                pool = stack.enter_context(mp.get_context("spawn").Pool(n_workers))
                results = pool.imap(_encode_one_video_frames, tasks, chunksize=1)
            reader = pa.RecordBatchReader.from_batches(_VIDEO_FRAME_SCHEMA, batch_stream(results))
            dataset = lance.write_dataset(reader, cache_dir, mode="overwrite", storage_options=storage_options)

            total_frames = sum(int(video["T"]) for video in video_records)
            meta = {
                "cache_format_version": _CACHE_FORMAT_VERSION,
                "format": "lance",
                "layout": "lance-video-frames",
                "schema_fingerprint": _features_fingerprint(features),
                "num_rows": total_frames,
                "total_rows": total_frames,
                "n_videos": len(video_records),
                "n_skipped": len(tasks) - len(video_records),
                "encoding": "webp",
                "quality": int(quality),
                "resize": int(resize) if resize is not None else None,
                "video_column": video_column,
                "videos": video_records,
                "version": dataset.version,
            }
            if lineage:
                meta["lineage"] = lineage
            _uri.write_text(_uri.join(cache_dir, _METADATA_FILE), json.dumps(meta, indent=2), storage_options)
    finally:
        shutil.rmtree(input_tmp_dir, ignore_errors=True)

    logging.info(
        f"Cached {meta['total_rows']} video frames from {meta['n_videos']} videos in Lance format to {cache_dir}"
    )
    return LanceCacheMeta(
        cache_dir=cache_dir,
        num_rows=meta["total_rows"],
        schema_fingerprint=meta["schema_fingerprint"],
        version=meta["version"],
    )


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


def detect_cache_format(cache_dir: str | Path, storage_options: dict | None = None) -> str:
    """Return ``"arrow"`` or ``"lance"`` based on the cache's metadata."""
    return _metadata_field(_read_metadata(cache_dir, storage_options), "format", cache_dir)


def detect_cache_layout(cache_dir: str | Path, storage_options: dict | None = None) -> str:
    """Return the physical cache layout recorded in cache metadata."""
    return _metadata_field(_read_metadata(cache_dir, storage_options), "layout", cache_dir)


class CacheOpenResult:
    """Result of opening cache metadata into a backend."""

    __slots__ = ("backend", "num_rows", "layout", "metadata")

    def __init__(self, *, backend, num_rows: int, layout: str, metadata):
        self.backend = backend
        self.num_rows = int(num_rows)
        self.layout = layout
        self.metadata = metadata


def open_cache(
    cache_dir: str | Path,
    features: Features,
    *,
    backend_kwargs: dict | None = None,
    storage_options: dict | None = None,
) -> CacheOpenResult:
    """Open a cache directory and return the backend selected by its layout.

    ``cache_dir`` is a local path or a URI. Lance caches can be remote.
    Arrow caches must be local. ``backend_kwargs`` go to the backend and
    replace the values from the metadata. ``storage_options`` go to Lance
    and to the metadata reads.
    """
    backend_kwargs = backend_kwargs or {}
    cache_dir = os.fspath(cache_dir)
    layout = detect_cache_layout(cache_dir, storage_options)

    if layout == "arrow-shards":
        from .backends.arrow_shards import ArrowBackend

        meta = validate_sharded_cache(cache_dir, features)
        backend = ArrowBackend(
            shard_paths=meta.shard_paths,
            shard_row_counts=meta.shard_row_counts,
            schema=features.to_arrow_schema(),
        )
        return CacheOpenResult(backend=backend, num_rows=meta.num_rows, layout=layout, metadata=meta)

    if layout == "lance-rows":
        from .backends.lance_rows import LanceBackend

        meta = read_lance_cache_meta(cache_dir, storage_options)
        _check_schema_fingerprint(meta.schema_fingerprint, features, cache_dir)
        defaults = {
            "shard_row_counts": meta.shard_row_counts,
            "storage_options": storage_options,
            "version": meta.version,
        }
        other_version = backend_kwargs.get("version", meta.version) != meta.version
        if other_version:
            # The row counts in the metadata belong to the metadata version.
            defaults["shard_row_counts"] = None
        backend = LanceBackend(uri=cache_dir, **{**defaults, **backend_kwargs})
        num_rows = backend.num_rows if other_version else meta.num_rows
        return CacheOpenResult(backend=backend, num_rows=num_rows, layout=layout, metadata=meta)

    if layout == "lance-video-frames":
        from .backends.lance_video_frames import LanceVideoFramesBackend

        backend = LanceVideoFramesBackend(uri=cache_dir, **{"storage_options": storage_options, **backend_kwargs})
        _check_format_version(backend.metadata)
        _check_schema_fingerprint(backend.metadata["schema_fingerprint"], features, cache_dir)
        return CacheOpenResult(backend=backend, num_rows=backend.num_rows, layout=layout, metadata=backend.metadata)

    raise ValueError(f"Unknown cache layout {layout!r} in {cache_dir}")


def read_lance_cache_meta(cache_dir: str | Path, storage_options: dict | None = None) -> LanceCacheMeta:
    """Read metadata from a Lance-format cache directory.

    Returns a :class:`LanceCacheMeta` with the cached row count and
    schema fingerprint populated from ``_metadata.json``. Deliberately
    does NOT open the underlying Lance dataset: that would initialize
    Lance's tokio runtime in the caller's process, which is a
    DataLoader-fork footgun. Row count comes from the metadata file,
    not from ``ds.count_rows()``.
    """
    raw = _read_metadata(cache_dir, storage_options)
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


class ShardedCacheMeta:
    """Lightweight descriptor for a sharded Arrow cache on disk."""

    __slots__ = (
        "cache_dir",
        "num_rows",
        "num_shards",
        "shard_filenames",
        "shard_row_counts",
        "schema_fingerprint",
        "compression",
    )

    def __init__(
        self,
        cache_dir: Path,
        num_rows: int,
        num_shards: int,
        shard_filenames: list[str],
        shard_row_counts: list[int],
        schema_fingerprint: str,
        compression: str | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.num_rows = num_rows
        self.num_shards = num_shards
        self.shard_filenames = shard_filenames
        self.shard_row_counts = shard_row_counts
        self.schema_fingerprint = schema_fingerprint
        self.compression = compression

    @property
    def shard_paths(self) -> list[Path]:
        return [self.cache_dir / f for f in self.shard_filenames]


def read_sharded_cache_meta(cache_dir: Path) -> ShardedCacheMeta:
    """Read metadata from a sharded cache directory.

    Validates that all shard files and metadata exist and are internally
    consistent.  Raises ``ValueError`` on corruption. Arrow caches are
    memory-mapped, so ``cache_dir`` must be local.
    """
    local_dir = _uri.local_path(cache_dir)
    if local_dir is None:
        raise ValueError(f"Arrow caches must be local, got {cache_dir!r}. Use a Lance cache for a remote URI.")
    cache_dir = local_dir
    raw = _read_metadata(cache_dir)

    if raw.get("format") != "arrow":
        raise ValueError(f"Not an Arrow sharded cache: {cache_dir} (format={raw.get('format')!r})")
    if raw.get("layout") != "arrow-shards":
        raise ValueError(f"Not an Arrow sharded cache: {cache_dir} (layout={raw.get('layout')!r})")

    _check_format_version(raw)

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
        compression=raw.get("compression"),
    )


def validate_sharded_cache(cache_dir: Path, features: Features) -> ShardedCacheMeta:
    """Read and validate a sharded cache, checking the schema fingerprint.

    Raises ``ValueError`` if the cache is inconsistent or the schema has changed.
    """
    meta = read_sharded_cache_meta(cache_dir)
    _check_schema_fingerprint(meta.schema_fingerprint, features, cache_dir)
    return meta


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


def read_shard(shard_path: Path) -> pa.Table:
    """Memory-map a single shard file and return its table."""
    mmap = pa.memory_map(str(shard_path), "r")
    reader = ipc.open_file(mmap)
    return reader.read_all()
