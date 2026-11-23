"""Generator-to-Lance caching pipeline.

Writes dataset examples to a Lance dataset (``lance-rows``), or videos to
the frame layout (``lance-video-frames``), and opens a cache by the layout
its ``_metadata.json`` records. Peak memory during writes is bounded to
about one batch.
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

import numpy as np
import pyarrow as pa
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


# Writers

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

    The generator yields ``(key, example)`` pairs; examples are encoded by
    the features and stream into ``lance.write_dataset`` through a
    :class:`pa.RecordBatchReader`.

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
        .
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
    cache_dir = os.fspath(cache_dir)
    schema = features.to_arrow_schema()
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

    return _write_lance_batches(
        batch_stream(),
        features,
        cache_dir,
        max_rows_per_file=max_rows_per_file,
        max_bytes_per_file=max_bytes_per_file,
        lineage=lineage,
        storage_options=storage_options,
    )


def write_lance_table(
    table: pa.Table,
    features: Features,
    cache_dir: str | Path,
    *,
    lineage: dict | None = None,
    storage_options: dict | None = None,
) -> LanceCacheMeta:
    """Write rows that are already encoded (a table read from another cache) as a Lance cache."""
    table = table.cast(features.to_arrow_schema())
    return _write_lance_batches(
        iter(table.to_batches(max_chunksize=1000)),
        features,
        os.fspath(cache_dir),
        lineage=lineage,
        storage_options=storage_options,
    )


def _write_lance_batches(
    batches,
    features: Features,
    cache_dir: str,
    *,
    max_rows_per_file: int = DEFAULT_MAX_ROWS_PER_FILE,
    max_bytes_per_file: int = DEFAULT_SHARD_SIZE_BYTES,
    lineage: dict | None = None,
    storage_options: dict | None = None,
) -> LanceCacheMeta:
    """Write record batches as the Lance dataset at ``cache_dir``, then its ``_metadata.json``."""
    import lance

    schema = features.to_arrow_schema()
    with _writer_lock(cache_dir):
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
        rows = dataset.count_rows()
        meta = {
            "cache_format_version": _CACHE_FORMAT_VERSION,
            "format": "lance",
            "layout": "lance-rows",
            "schema_fingerprint": _features_fingerprint(features),
            "num_rows": rows,
            "shard_row_counts": [fragment.count_rows() for fragment in dataset.get_fragments()],
            "version": dataset.version,
        }
        if lineage:
            meta["lineage"] = lineage
        _uri.write_text(_uri.join(cache_dir, _METADATA_FILE), json.dumps(meta, indent=2), storage_options)

    logging.info(f"Cached {rows} examples in Lance format to {cache_dir}")

    return LanceCacheMeta(
        cache_dir=cache_dir,
        num_rows=rows,
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

        if workers:
            n_workers = int(workers)
        else:  # the CPUs this process may use (os.cpu_count() is the host's, inside a container)
            from .framestore._native import usable_cpus

            n_workers = usable_cpus()
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


def write_framestore_video_cache(
    generator,
    features: Features,
    cache_dir: str | Path,
    *,
    video_column: str | None = None,
    cpu: float | int = 0.8,
    skip_corrupt: bool = True,
    lineage: dict | None = None,
    storage_options: dict | None = None,
) -> LanceCacheMeta:
    """Write a framestore video cache: the video column's VideoStorage (``Video(storage="mp4" | "jpeg")``) built by
    stable_datasets.framestore.build_store, one episode per source video, the other scalar fields per video. The
    store is the cache directory's Lance dataset; ``_metadata.json`` records each video's frame count, key and
    fields, which the backend's clip windows come from. Local caches only (the store is read by memory map)."""
    import lance

    from .framestore.build import build_store

    stores = [n for n, f in features.items() if isinstance(f, Video) and f.storage in Video.STORES]
    video_column = video_column or (stores[0] if len(stores) == 1 else None)
    if video_column is None:
        raise ValueError("write_framestore_video_cache needs exactly one Video(storage='mp4' | 'jpeg') column")
    feat = features[video_column]
    vs = feat.settings
    cache_dir = os.fspath(cache_dir)
    local = _uri.local_path(cache_dir)
    if local is None:
        raise ValueError(f"a framestore video cache is local (memory-mapped stores); {cache_dir} is remote")
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=local.parent, prefix=f".{local.name}_inputs_"))
    try:
        paths, examples = [], []
        for key, example in generator:
            path, checksum = _video_input_to_path(
                example[video_column], tmp_dir=tmp, allowed_extensions=feat.allowed_extensions
            )
            fields = {
                k: v for k, v in example.items() if k != video_column and isinstance(v, str | int | float | bool)
            }
            paths.append(str(path))
            examples.append({"key": key if isinstance(key, str | int | float | bool) else str(key), "checksum": checksum,
                             "path": str(path), "metadata": fields})  # fmt: skip
        if not paths:
            raise ValueError("Cannot build a framestore video cache from an empty generator.")
        with _writer_lock(cache_dir):
            if local.exists() and not (local / _METADATA_FILE).exists():
                shutil.rmtree(local)  # a build interrupted before its metadata: start over
            stats = build_store(
                paths, cache_dir, short_edge=vs.short_edge, fps=vs.fps, quality=vs.quality, format=vs.format,
                crf=vs.crf, keyint=vs.keyint, canvas=vs.canvas, cpu=cpu, log=logging.info,
            )  # fmt: skip
            if stats["failed"] and not skip_corrupt:
                raise RuntimeError(f"failed on {stats['failed'][0][0]}: {stats['failed'][0][1]}")
            ds = lance.dataset(cache_dir)
            if vs.format == "mp4":
                t = ds.to_table(columns=["episode_idx", "frames"])
                frames = dict(zip(t.column(0).to_pylist(), t.column(1).to_pylist()))
            else:
                ep = ds.to_table(columns=["episode_idx"]).column(0).to_numpy()
                u, c = np.unique(ep, return_counts=True)
                frames = dict(zip(u.tolist(), c.tolist()))
            # store episodes in store order (row order: what the readers index), each with its input's key and fields
            order = (
                ds.to_table(columns=["episode_idx"]).column(0).to_pylist()
                if vs.format == "mp4"
                else list(dict.fromkeys(ds.to_table(columns=["episode_idx"]).column(0).to_pylist()))
            )
            videos = [{"episode": int(e), "T": int(frames[e]), **examples[e]} for e in order]
            meta = {
                "cache_format_version": _CACHE_FORMAT_VERSION,
                "format": "lance",
                "layout": "framestore-video",
                "schema_fingerprint": _features_fingerprint(features),
                "store_format": vs.format,
                "settings": {
                    k: getattr(vs, k) for k in ("format", "fps", "short_edge", "crf", "keyint", "canvas", "quality")
                },
                "video_column": video_column,
                "n_videos": len(videos),
                "n_skipped": len(stats["failed"]),
                "videos": videos,
                "version": ds.version,
            }
            if lineage:
                meta["lineage"] = lineage
            _uri.write_text(_uri.join(cache_dir, _METADATA_FILE), json.dumps(meta, indent=2), storage_options)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    logging.info(f"Built a {vs.format} store of {len(videos)} videos at {cache_dir}")
    return LanceCacheMeta(cache_dir=cache_dir, num_rows=len(videos), schema_fingerprint=meta["schema_fingerprint"],
                          version=meta["version"])  # fmt: skip


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

    ``cache_dir`` is a local path or a URI. ``backend_kwargs`` go to the backend and
    replace the values from the metadata. ``storage_options`` go to Lance
    and to the metadata reads.
    """
    backend_kwargs = backend_kwargs or {}
    cache_dir = os.fspath(cache_dir)
    layout = detect_cache_layout(cache_dir, storage_options)

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

    if layout == "framestore-video":
        from .backends.framestore_video import FramestoreVideoBackend

        backend = FramestoreVideoBackend(uri=cache_dir, **{"storage_options": storage_options, **backend_kwargs})
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
