"""The video-frames layout (``lance-video-frames``): one WebP-encoded frame per Lance row, one video per example."""

from __future__ import annotations

import contextlib
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
from loguru import logger as logging

from .. import _uri
from ..cache import (
    _CACHE_FORMAT_VERSION,
    _METADATA_FILE,
    CacheOpenResult,
    LanceCacheMeta,
    _check_format_version,
    _check_schema_fingerprint,
    _features_fingerprint,
    _writer_lock,
)
from ..schema import Features, Video
from .base import Storage, register_layout


@register_layout
@dataclass(frozen=True)
class VideoFramesStorage(Storage):
    """The video column stored as WebP frames, one Lance row per frame (``Video(storage="frames")``)."""

    layout = "lance-video-frames"
    aliases = {"frames": {}}

    @classmethod
    def from_features(cls, features: Features) -> VideoFramesStorage | None:
        if any(isinstance(f, Video) and f.storage == "frames" for f in features.values()):
            return cls()
        return None

    def apply(self, features: Features) -> Features:
        return Features(
            (name, Video(storage="frames", allowed_extensions=f.allowed_extensions)) if isinstance(f, Video) else (name, f)
            for name, f in features.items()
        )  # fmt: skip

    def write(self, generator, features, cache_dir, *, lineage=None, storage_options=None) -> LanceCacheMeta:
        return write_lance_video_frames_cache(
            generator, features, cache_dir, lineage=lineage, storage_options=storage_options
        )

    @classmethod
    def open(cls, cache_dir, features, metadata, *, backend_kwargs, storage_options=None) -> CacheOpenResult:
        from ..backends.lance_video_frames import LanceVideoFramesBackend

        backend = LanceVideoFramesBackend(uri=cache_dir, **{"storage_options": storage_options, **backend_kwargs})
        _check_format_version(backend.metadata)
        _check_schema_fingerprint(backend.metadata["schema_fingerprint"], features, cache_dir)
        return CacheOpenResult(
            backend=backend, num_rows=backend.num_rows, layout=cls.layout, metadata=backend.metadata
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
            from ..framestore._native import usable_cpus

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
