"""One-shot converter: sharded Arrow IPC cache -> Lance dataset.

**Research scaffold, not the production pipeline.** This tool exists so
we can test :class:`LanceBackend` against real existing caches without
first building a parallel ``write_lance_cache`` generator path. It reads
the ``shard-NNNNN.arrow`` files from a stable-datasets cache directory
and streams them into ``lance.write_dataset`` as a ``RecordBatchReader``
(no intermediate materialization), upcasting ``binary`` columns to
``large_binary`` on the fly.

Workflow today (Phase A):

    builder downloads + encodes data
        -> write_sharded_arrow_cache (Arrow IPC shards)
        -> arrow_to_lance (this tool)    [throwaway conversion step]
        -> LanceBackend reads the Lance copy

Workflow once the Lance backend is validated (Phase C):

    builder downloads + encodes data
        -> write_lance_cache (direct streaming write)
        -> LanceBackend reads the Lance dataset

The conversion step disappears entirely. Nothing downstream depends on
it; it's throwaway scaffolding that lets us reach a working end-to-end
Lance path cheaply enough to profile before committing to a new writer.

CLI:

    python -m stable_datasets.tools.arrow_to_lance \\
        --arrow-cache /path/to/imagenet1k_default_train_... \\
        --out /path/to/imagenet1k_lance
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc


def _promote_binary_schema(schema: pa.Schema) -> pa.Schema:
    fields = []
    for f in schema:
        if pa.types.is_binary(f.type) and not pa.types.is_large_binary(f.type):
            fields.append(pa.field(f.name, pa.large_binary(), metadata=f.metadata))
        else:
            fields.append(f)
    return pa.schema(fields)


def _shard_paths_from_meta(cache_dir: Path) -> list[Path]:
    meta_path = cache_dir / "_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Not a stable-datasets cache: {cache_dir} (no _metadata.json)")
    meta = json.loads(meta_path.read_text())
    return [cache_dir / name for name in meta["shard_filenames"]]


def _iter_batches_across_shards(shard_paths: list[Path], target_schema: pa.Schema):
    """Yield ``pa.RecordBatch`` from every shard, casting binary columns
    to ``large_binary`` to match ``target_schema``."""
    for path in shard_paths:
        mmap = pa.memory_map(str(path), "r")
        reader = ipc.open_file(mmap)
        src = reader.schema
        cast_names = [
            f.name
            for f in target_schema
            if src.field(f.name).type != f.type
        ]
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            if cast_names:
                arrays = []
                for f in target_schema:
                    col = batch.column(f.name)
                    if f.name in cast_names:
                        col = col.cast(f.type)
                    arrays.append(col)
                yield pa.record_batch(arrays, schema=target_schema)
            else:
                yield batch
        del reader, mmap


def arrow_to_lance(
    arrow_cache: Path,
    out_dir: Path,
    *,
    overwrite: bool = False,
) -> Path:
    import lance

    arrow_cache = Path(arrow_cache)
    out_dir = Path(out_dir)

    if out_dir.exists():
        if overwrite:
            shutil.rmtree(out_dir)
        else:
            raise FileExistsError(
                f"{out_dir} already exists; pass overwrite=True to replace"
            )

    shard_paths = _shard_paths_from_meta(arrow_cache)
    # Peek at first shard's schema, promote binary -> large_binary.
    first = pa.memory_map(str(shard_paths[0]), "r")
    src_schema = ipc.open_file(first).schema
    del first
    target_schema = _promote_binary_schema(src_schema)

    rbr = pa.RecordBatchReader.from_batches(
        target_schema, _iter_batches_across_shards(shard_paths, target_schema)
    )

    t0 = time.perf_counter()
    lance.write_dataset(rbr, str(out_dir))
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrow-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    print(f"Converting Arrow cache -> Lance:")
    print(f"  src: {args.arrow_cache}")
    print(f"  dst: {args.out}")
    t0 = time.perf_counter()
    arrow_to_lance(args.arrow_cache, args.out, overwrite=args.overwrite)
    print(f"  done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
