"""Lance-backed random-access video frame segment layout."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from .. import _uri


_LANCE_CACHE: dict[tuple[str, int], Any] = {}

# Windows per record batch in iter_batches. One Lance take reads all frames of a batch.
_WINDOWS_PER_BATCH = 64


def _open_dataset(uri: str, version: int, storage_options: dict | None):
    ds = _LANCE_CACHE.get((uri, version))
    if ds is None:
        import lance

        ds = lance.dataset(uri, version=version, storage_options=storage_options)
        _LANCE_CACHE[(uri, version)] = ds
    return ds


def reset_worker_state() -> None:
    """Reset process-local decoder/backend state after a DataLoader fork."""
    global _LANCE_CACHE
    _LANCE_CACHE = {}
    try:
        import cv2

        cv2.setNumThreads(1)
    except Exception:
        pass


def _compute_segment_plan(
    videos: list[dict],
    window_length: int,
    frame_skip: int,
    hop_size: int,
    min_video_frames: int | None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    if window_length < 1:
        raise ValueError(f"window_length must be >= 1, got {window_length}")
    if frame_skip < 0:
        raise ValueError(f"frame_skip must be >= 0, got {frame_skip}")
    if hop_size < 1:
        raise ValueError(f"hop_size must be >= 1, got {hop_size}")

    stride = frame_skip + 1
    span = (window_length - 1) * stride + 1
    min_t = max(span, int(min_video_frames) if min_video_frames is not None else span)

    seg_vid: list[int] = []
    seg_start: list[int] = []
    per_video: list[int] = []
    for vi, video in enumerate(videos):
        frames = int(video["T"])
        if frames < min_t:
            per_video.append(0)
            continue
        last_start = frames - span
        n = last_start // hop_size + 1
        per_video.append(int(n))
        starts = range(0, last_start + 1, hop_size)
        seg_vid.extend([vi] * n)
        seg_start.extend(starts)
    return np.asarray(seg_vid, dtype=np.int64), np.asarray(seg_start, dtype=np.int64), per_video


class LanceVideoFramesBackend:
    """StorageBackend for the ``lance-video-frames`` layout.

    The physical Lance dataset stores one WebP-encoded frame per row. The
    logical dataset exposes deterministic frame windows as samples.

    Each row holds the encoded frames of one window in the video column.
    The ``Video(storage="frames")`` feature decodes them. Each video with
    at least one window is one shard.
    """

    prefer_batched_take: bool = True

    def __init__(
        self,
        *,
        uri: str | Path,
        window_length: int = 1,
        frame_skip: int = 0,
        hop_size: int = 1,
        min_video_frames: int | None = None,
        batch_readahead: int = 8,
        storage_options: dict | None = None,
    ):
        meta_path = _uri.join(uri, "_metadata.json")
        if not _uri.exists(meta_path, storage_options):
            raise FileNotFoundError(f"No metadata file at {meta_path}")
        self._configure(
            uri=uri,
            metadata=json.loads(_uri.read_text(meta_path, storage_options)),
            window_length=window_length,
            frame_skip=frame_skip,
            hop_size=hop_size,
            min_video_frames=min_video_frames,
            batch_readahead=batch_readahead,
            storage_options=storage_options,
        )

    def _configure(
        self,
        *,
        uri: str | Path,
        metadata: dict,
        window_length: int,
        frame_skip: int,
        hop_size: int,
        min_video_frames: int | None,
        batch_readahead: int,
        storage_options: dict | None,
        plan: tuple[np.ndarray, np.ndarray, list[int]] | None = None,
    ) -> None:
        self._uri = os.fspath(uri)
        self._storage_options = storage_options
        self._batch_readahead = int(batch_readahead)
        self.window_length = int(window_length)
        self.frame_skip = int(frame_skip)
        self.hop_size = int(hop_size)
        self.min_video_frames = min_video_frames
        self._stride = self.frame_skip + 1
        self._span = (self.window_length - 1) * self._stride + 1

        self.metadata = metadata
        self._column = metadata["video_column"]
        self._videos = list(metadata.get("videos", []))
        if not self._videos:
            raise ValueError(f"no videos recorded in {_uri.join(self._uri, '_metadata.json')}")
        self._start_rows = np.asarray([video["start_row"] for video in self._videos], dtype=np.int64)
        self._metadata_keys = list(dict.fromkeys(k for video in self._videos for k in video.get("metadata") or {}))

        if plan is None:
            plan = _compute_segment_plan(
                self._videos,
                self.window_length,
                self.frame_skip,
                self.hop_size,
                min_video_frames,
            )
        self._seg_vid, self._seg_start, self._per_video = plan
        if self._seg_vid.size == 0:
            raise ValueError(f"no valid segments: every video has fewer than span={self._span} frames")
        self._shard_row_counts = [n for n in self._per_video if n > 0]
        self._shard_starts = np.cumsum([0, *self._shard_row_counts])
        self._schema: pa.Schema | None = None
        self._ds = None

    def with_window(self, *, window_length: int, frame_skip: int) -> LanceVideoFramesBackend:
        """Return a copy that uses a different window length and frame skip."""
        backend = object.__new__(type(self))
        backend._configure(
            uri=self._uri,
            metadata=self.metadata,
            window_length=window_length,
            frame_skip=frame_skip,
            hop_size=self.hop_size,
            min_video_frames=self.min_video_frames,
            batch_readahead=self._batch_readahead,
            storage_options=self._storage_options,
        )
        return backend

    @property
    def _dataset(self):
        if self._ds is None:
            # Read the Lance version that the metadata describes.
            self._ds = _open_dataset(self._uri, self.metadata["version"], self._storage_options)
        return self._ds

    @property
    def num_rows(self) -> int:
        return int(self._seg_vid.shape[0])

    @property
    def num_shards(self) -> int:
        return len(self._shard_row_counts)

    @property
    def shard_row_counts(self) -> list[int]:
        return list(self._shard_row_counts)

    @property
    def is_file_backed(self) -> bool:
        return True

    @property
    def cache_dir(self) -> Path | None:
        """Local cache directory, or ``None`` for a remote URI."""
        return _uri.local_path(self._uri)

    @property
    def schema(self) -> pa.Schema:
        """Schema of the window rows. Build it from the metadata, not from Lance."""
        if self._schema is None:
            fields = [
                pa.field(self._column, pa.list_(pa.large_binary())),
                pa.field("video_idx", pa.int64()),
                pa.field("filename", pa.string()),
                pa.field("start_frame", pa.int64()),
                pa.field("frame_indices", pa.list_(pa.int64())),
                pa.field("sample_idx", pa.int64()),
            ]
            for key in self._metadata_keys:
                values = [(video.get("metadata") or {}).get(key) for video in self._videos]
                fields.append(pa.field(key, pa.array(values).type))
            self._schema = pa.schema(fields)
        return self._schema

    @property
    def table(self) -> pa.Table:
        return self.take(np.arange(self.num_rows))

    @property
    def video_paths(self) -> list[str]:
        return [video["path"] for video in self._videos]

    @property
    def segment_filenames(self) -> list[str]:
        paths = self.video_paths
        return [paths[int(video_id)] for video_id in self._seg_vid]

    def segment_filename(self, idx: int) -> str:
        return self._videos[int(self._seg_vid[idx])]["path"]

    def segment_info(self, idx: int) -> dict:
        vi = int(self._seg_vid[idx])
        start = int(self._seg_start[idx])
        return {
            "video_idx": vi,
            "filename": self._videos[vi]["path"],
            "start_frame": start,
            "frame_indices": [start + j * self._stride for j in range(self.window_length)],
        }

    def get_row(self, idx: int) -> dict:
        if idx < 0:
            idx += self.num_rows
        if idx < 0 or idx >= self.num_rows:
            raise IndexError(idx)
        return self.take([idx]).to_pylist()[0]

    def take(self, indices: np.ndarray | list[int]) -> pa.Table:
        """Return the windows at ``indices``. Read all their frames with one Lance take."""
        segments = np.asarray(indices, dtype=np.int64)
        video_ids = self._seg_vid[segments]
        starts = self._seg_start[segments]
        frame_indices = starts[:, None] + np.arange(self.window_length, dtype=np.int64) * self._stride
        frame_rows = (self._start_rows[video_ids][:, None] + frame_indices).ravel()

        unique_rows, positions = np.unique(frame_rows, return_inverse=True)
        if unique_rows.size:
            blobs = self._dataset.take(unique_rows.tolist(), columns=["bytes"]).column("bytes").combine_chunks()
        else:
            blobs = pa.array([], type=pa.large_binary())
        offsets = pa.array(np.arange(0, frame_rows.size + 1, self.window_length, dtype=np.int32))

        columns = {
            self._column: pa.ListArray.from_arrays(offsets, blobs.take(pa.array(positions))),
            "video_idx": pa.array(video_ids),
            "filename": pa.array([self._videos[vi]["path"] for vi in video_ids.tolist()], type=pa.string()),
            "start_frame": pa.array(starts),
            "frame_indices": pa.ListArray.from_arrays(offsets, pa.array(frame_indices.ravel())),
            "sample_idx": pa.array(segments),
        }
        for key in self._metadata_keys:
            values = [(self._videos[vi].get("metadata") or {}).get(key) for vi in video_ids.tolist()]
            columns[key] = pa.array(values, type=self.schema.field(key).type)
        return pa.table(columns, schema=self.schema)

    def slice(self, start: int, length: int) -> pa.Table:
        return self.take(np.arange(start, start + length))

    def iter_batches(
        self,
        shard_indices: list[int] | None = None,
        shuffle: bool = False,
        seed: int | None = None,
    ):
        """Yield the windows of each shard (video) in record batches.

        With ``shuffle=True``, shuffle the shard order. A ``seed`` of
        ``None`` gives a different order on each call.
        """
        shards = list(range(self.num_shards)) if shard_indices is None else list(shard_indices)
        if shuffle:
            np.random.default_rng(seed).shuffle(shards)
        for shard_id in shards:
            lo, hi = int(self._shard_starts[shard_id]), int(self._shard_starts[shard_id + 1])
            for start in range(lo, hi, _WINDOWS_PER_BATCH):
                yield from self.take(np.arange(start, min(start + _WINDOWS_PER_BATCH, hi))).to_batches()

    def __getstate__(self) -> dict:
        return {
            "uri": self._uri,
            "storage_options": self._storage_options,
            "metadata": self.metadata,
            "window_length": self.window_length,
            "frame_skip": self.frame_skip,
            "hop_size": self.hop_size,
            "min_video_frames": self.min_video_frames,
            "batch_readahead": self._batch_readahead,
            "plan": (self._seg_vid, self._seg_start, self._per_video),
        }

    def __setstate__(self, state: dict) -> None:
        self._configure(**state)

    @staticmethod
    def worker_init(worker_id: int) -> None:
        del worker_id
        reset_worker_state()


__all__ = ["LanceVideoFramesBackend", "reset_worker_state"]
