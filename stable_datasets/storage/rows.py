"""The row layout (``lance-rows``): one Lance row per example, columns encoded by the features."""

from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
from loguru import logger as logging

from .. import _uri
from ..cache import (
    _CACHE_FORMAT_VERSION,
    _METADATA_FILE,
    DEFAULT_MAX_ROWS_PER_FILE,
    DEFAULT_SHARD_SIZE_BYTES,
    CacheOpenResult,
    LanceCacheMeta,
    _check_schema_fingerprint,
    _encode_gen,
    _features_fingerprint,
    _writer_lock,
    lance_cache_meta,
)
from ..schema import Features
from .base import Storage, register_layout


@register_layout
@dataclass(frozen=True)
class RowStorage(Storage):
    """One Lance row per example (the default)."""

    layout = "lance-rows"
    aliases = {"rows": {}}
    priority = 0

    @classmethod
    def from_features(cls, features: Features) -> RowStorage:
        return cls()

    def write(self, generator, features, cache_dir, *, lineage=None, storage_options=None) -> LanceCacheMeta:
        return write_lance_cache(
            generator, features, cache_dir, num_encode_workers=4, lineage=lineage, storage_options=storage_options
        )

    @classmethod
    def open(cls, cache_dir, features, metadata, *, backend_kwargs, storage_options=None) -> CacheOpenResult:
        from ..backends.lance_rows import LanceBackend

        meta = lance_cache_meta(metadata, cache_dir)
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
        return CacheOpenResult(backend=backend, num_rows=num_rows, layout=cls.layout, metadata=meta)


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
