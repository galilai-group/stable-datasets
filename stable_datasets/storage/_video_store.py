"""Writes the framestore video layout (VideoStorage.write)."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from loguru import logger as logging

from .. import _uri
from ..cache import _CACHE_FORMAT_VERSION, _METADATA_FILE, LanceCacheMeta, _features_fingerprint, _writer_lock
from ..schema import Features, Video
from .video_frames import _video_input_to_path


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

    from ..framestore.build import build_store

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
