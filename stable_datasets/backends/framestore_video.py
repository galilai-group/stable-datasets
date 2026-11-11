"""The ``framestore-video`` layout: a stable_datasets.framestore store (MP4 or JPEG) whose samples are clip windows."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa

from .. import _uri


_WINDOWS_PER_BATCH = 64


class FramestoreVideoBackend:
    """StorageBackend for a framestore video cache (``Video(storage="mp4" | "jpeg")``).

    The store holds one episode per video; the logical dataset is its clip windows of ``window_length`` frames,
    ``frame_skip + 1`` apart, starting every ``hop_size`` frames. In an MP4 store the frame stride was fixed when it
    was built (frame_skip must be 0) and windows start on keyframes (hop_size: the store's keyframe spacing, or a
    multiple of it). Each row's video column is a window reference ({"store", "format", "video", "start", "frames",
    "stride"}), decoded by the Video feature or by a dataset's video decode. Each video with a window is one shard.
    """

    prefer_batched_take: bool = True

    def __init__(
        self,
        *,
        uri: str | Path,
        window_length: int = 1,
        frame_skip: int = 0,
        hop_size: int | None = None,
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
            storage_options=storage_options,
        )

    def _configure(self, *, uri, metadata, window_length, frame_skip, hop_size, storage_options):
        self._uri = os.fspath(uri)
        self._storage_options = storage_options
        self.metadata = metadata
        self.format = metadata["store_format"]
        self._column = metadata["video_column"]
        self._videos = list(metadata["videos"])
        keyint = int(metadata["settings"]["keyint"]) if self.format == "mp4" else 1
        if self.format == "mp4" and frame_skip:
            raise ValueError(
                "an MP4 store keeps frames at its build's fps (VideoStorage.fps): frame_skip must be 0 "
                "(frame_stride 1); build a JPEG store for strides chosen when loading"
            )
        hop = keyint if hop_size is None else int(hop_size)
        if hop % keyint:
            raise ValueError(f"windows of an MP4 store start on keyframes: hop_size must be a multiple of {keyint}")
        self.window_length, self.frame_skip, self.hop_size = int(window_length), int(frame_skip), hop
        stride = self.frame_skip + 1
        span = (self.window_length - 1) * stride + 1
        vid, start, per = [], [], []
        for i, v in enumerate(self._videos):
            n = (int(v["T"]) - span) // hop + 1 if int(v["T"]) >= span else 0
            per.append(n)
            vid += [i] * n
            start += list(range(0, n * hop, hop))
        if not vid:
            raise ValueError(f"no windows: every video has fewer than {span} frames")
        self._vid = np.asarray(vid, np.int64)
        self._start = np.asarray(start, np.int64)
        self._shards = [n for n in per if n]
        self._shard_starts = np.cumsum([0, *self._shards])
        self._keys = list(dict.fromkeys(k for v in self._videos for k in v.get("metadata") or {}))
        self._schema = None

    def with_window(self, *, window_length: int, frame_skip: int) -> FramestoreVideoBackend:
        backend = object.__new__(type(self))
        backend._configure(
            uri=self._uri,
            metadata=self.metadata,
            window_length=window_length,
            frame_skip=frame_skip,
            hop_size=None,
            storage_options=self._storage_options,
        )
        return backend

    @property
    def store_path(self) -> str:
        return self._uri

    @property
    def num_rows(self) -> int:
        return int(self._vid.shape[0])

    @property
    def num_shards(self) -> int:
        return len(self._shards)

    @property
    def shard_row_counts(self) -> list[int]:
        return list(self._shards)

    @property
    def is_file_backed(self) -> bool:
        return True

    @property
    def cache_dir(self) -> Path | None:
        return _uri.local_path(self._uri)

    @property
    def schema(self) -> pa.Schema:
        if self._schema is None:
            ref = pa.struct(
                [
                    pa.field("store", pa.string()),
                    pa.field("format", pa.string()),
                    pa.field("video", pa.int64()),
                    pa.field("start", pa.int64()),
                    pa.field("frames", pa.int64()),
                    pa.field("stride", pa.int64()),
                ]
            )
            fields = [
                pa.field(self._column, ref),
                pa.field("video_idx", pa.int64()),
                pa.field("start_frame", pa.int64()),
            ]
            for k in self._keys:
                fields.append(pa.field(k, pa.array([(v.get("metadata") or {}).get(k) for v in self._videos]).type))
            self._schema = pa.schema(fields)
        return self._schema

    @property
    def table(self) -> pa.Table:
        return self.take(np.arange(self.num_rows))

    def get_row(self, idx: int) -> dict:
        if idx < 0:
            idx += self.num_rows
        if not 0 <= idx < self.num_rows:
            raise IndexError(idx)
        return self.take([idx]).to_pylist()[0]

    def take(self, indices) -> pa.Table:
        rows = np.asarray(indices, np.int64)
        vids, starts = self._vid[rows], self._start[rows]
        n = len(rows)
        ref = pa.StructArray.from_arrays(
            [
                pa.array([self._uri] * n, pa.string()),
                pa.array([self.format] * n, pa.string()),
                pa.array(vids),
                pa.array(starts),
                pa.array(np.full(n, self.window_length, np.int64)),
                pa.array(np.full(n, self.frame_skip + 1, np.int64)),
            ],
            fields=list(self.schema.field(self._column).type),
        )
        cols = {self._column: ref, "video_idx": pa.array(vids), "start_frame": pa.array(starts)}
        for k in self._keys:
            cols[k] = pa.array(
                [(self._videos[v].get("metadata") or {}).get(k) for v in vids.tolist()], self.schema.field(k).type
            )
        return pa.table(cols, schema=self.schema)

    def slice(self, start: int, length: int) -> pa.Table:
        return self.take(np.arange(start, start + length))

    def iter_batches(self, shard_indices=None, shuffle: bool = False, seed: int | None = None):
        shards = list(range(self.num_shards)) if shard_indices is None else list(shard_indices)
        if shuffle:
            np.random.default_rng(seed).shuffle(shards)
        for s in shards:
            lo, hi = int(self._shard_starts[s]), int(self._shard_starts[s + 1])
            for a in range(lo, hi, _WINDOWS_PER_BATCH):
                yield from self.take(np.arange(a, min(a + _WINDOWS_PER_BATCH, hi))).to_batches()

    def __getstate__(self) -> dict:
        return {
            "uri": self._uri,
            "metadata": self.metadata,
            "window_length": self.window_length,
            "frame_skip": self.frame_skip,
            "hop_size": self.hop_size,
            "storage_options": self._storage_options,
        }

    def __setstate__(self, state: dict) -> None:
        self._configure(**state)


__all__ = ["FramestoreVideoBackend"]
