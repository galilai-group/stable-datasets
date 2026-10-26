"""Lance-backed random-access video frame segment layout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa


_FRAME_SCHEMA = pa.schema(
    [
        ("video_id", pa.int32()),
        ("frame_idx", pa.int32()),
        ("bytes", pa.large_binary()),
    ]
)

_LANCE_CACHE: dict[str, Any] = {}


def _open_dataset(path: str):
    ds = _LANCE_CACHE.get(path)
    if ds is None:
        import lance

        ds = lance.dataset(path)
        _LANCE_CACHE[path] = ds
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
    """

    # This backend supports a direct batched row path (get_rows) used by
    # StableDataset.__getitems__, so batched loading amortizes storage calls
    # without forcing expensive Arrow table materialization of video tensors.
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
    ):
        self._uri = Path(uri)
        self._batch_readahead = int(batch_readahead)
        self.window_length = int(window_length)
        self.frame_skip = int(frame_skip)
        self.hop_size = int(hop_size)
        self._stride = self.frame_skip + 1
        self._span = (self.window_length - 1) * self._stride + 1

        meta_path = self._uri / "_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No metadata file at {meta_path}")
        self.metadata = json.loads(meta_path.read_text())
        self._videos = list(self.metadata.get("videos", []))
        if not self._videos:
            raise ValueError(f"no videos recorded in {meta_path}")

        self._seg_vid, self._seg_start, self._per_video = _compute_segment_plan(
            self._videos,
            self.window_length,
            self.frame_skip,
            self.hop_size,
            min_video_frames,
        )
        if self._seg_vid.size == 0:
            raise ValueError(f"no valid segments: every video has fewer than span={self._span} frames")
        self._ds = None
        self._iter_batch_size = max(1, self._batch_readahead * 32)

    @property
    def _dataset(self):
        if self._ds is None:
            self._ds = _open_dataset(str(self._uri))
        return self._ds

    @property
    def num_rows(self) -> int:
        return int(self._seg_vid.shape[0])

    @property
    def num_shards(self) -> int:
        return 1

    @property
    def is_file_backed(self) -> bool:
        return True

    @property
    def cache_dir(self) -> Path:
        return self._uri

    @property
    def schema(self) -> pa.Schema:
        return _FRAME_SCHEMA

    @property
    def table(self) -> pa.Table:
        return self._dataset.to_table()

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
        return self._rows_for_indices([idx])[0]

    def get_rows(self, indices: np.ndarray | list[int]) -> list[dict]:
        return self._rows_for_indices(indices)

    def take(self, indices: np.ndarray | list[int]) -> pa.Table:
        rows = self._rows_for_indices(indices)
        serializable = []
        for row in rows:
            converted = dict(row)
            if hasattr(converted.get("video"), "tolist"):
                converted["video"] = converted["video"].tolist()
            serializable.append(converted)
        return pa.Table.from_pylist(serializable) if serializable else pa.table({})

    def slice(self, start: int, length: int) -> pa.Table:
        return self.take(list(range(start, start + length)))

    def iter_batches(
        self,
        shard_indices: list[int] | None = None,
        shuffle: bool = False,
        seed: int | None = None,
    ):
        if shard_indices is not None and 0 not in shard_indices:
            return
        indices = np.arange(self.num_rows, dtype=np.int64)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)
        for start in range(0, len(indices), self._iter_batch_size):
            chunk = indices[start : start + self._iter_batch_size]
            yield from self.take(chunk).to_batches(max_chunksize=self._iter_batch_size)

    def __getstate__(self) -> dict:
        return {
            "uri": str(self._uri),
            "window_length": self.window_length,
            "frame_skip": self.frame_skip,
            "hop_size": self.hop_size,
            "batch_readahead": self._batch_readahead,
        }

    def __setstate__(self, state: dict) -> None:
        self.__init__(
            uri=state["uri"],
            window_length=state["window_length"],
            frame_skip=state.get("frame_skip", 0),
            hop_size=state.get("hop_size", 1),
            batch_readahead=state.get("batch_readahead", 8),
        )

    def _normalize_index(self, idx: int) -> int:
        idx = int(idx)
        if idx < 0:
            idx += self.num_rows
        if idx < 0 or idx >= self.num_rows:
            raise IndexError(idx)
        return idx

    def _rows_for_indices(self, indices: np.ndarray | list[int]) -> list[dict]:
        if isinstance(indices, np.ndarray):
            requested = [int(i) for i in indices.tolist()]
        else:
            requested = [int(i) for i in indices]
        if not requested:
            return []

        normalized = [self._normalize_index(i) for i in requested]
        requests_by_video: dict[int, list[tuple[int, int, list[int], list[int]]]] = {}

        for out_pos, idx in enumerate(normalized):
            vi = int(self._seg_vid[idx])
            start = int(self._seg_start[idx])
            frame_indices = [start + j * self._stride for j in range(self.window_length)]
            row0 = int(self._videos[vi]["start_row"])
            row_ids = [row0 + frame_idx for frame_idx in frame_indices]
            requests_by_video.setdefault(vi, []).append((out_pos, idx, frame_indices, row_ids))

        out: list[dict | None] = [None] * len(normalized)
        for vi, reqs in requests_by_video.items():
            ordered_unique_rows: list[int] = []
            seen_rows: set[int] = set()
            for _, _, _, row_ids in reqs:
                for row_id in row_ids:
                    if row_id not in seen_rows:
                        seen_rows.add(row_id)
                        ordered_unique_rows.append(row_id)

            blobs = self._dataset.take(ordered_unique_rows, columns=["bytes"]).column("bytes").to_pylist()
            decoded_frames = self._decode_blobs(
                blobs,
                int(self._videos[vi]["H"]),
                int(self._videos[vi]["W"]),
            )
            by_row = {row_id: decoded_frames[i] for i, row_id in enumerate(ordered_unique_rows)}

            video = self._videos[vi]
            metadata = video.get("metadata")
            for out_pos, sample_idx, frame_indices, row_ids in reqs:
                frames = np.stack([by_row[row_id] for row_id in row_ids], axis=0)
                sample = {
                    "video": frames,
                    "video_idx": vi,
                    "filename": video["path"],
                    "start_frame": int(frame_indices[0]),
                    "frame_indices": frame_indices,
                    "sample_idx": int(sample_idx),
                }
                if isinstance(metadata, dict):
                    sample.update(metadata)
                out[out_pos] = sample

        return [row for row in out if row is not None]

    @staticmethod
    def worker_init(worker_id: int) -> None:
        del worker_id
        reset_worker_state()

    @staticmethod
    def _decode_blobs(blobs: list[bytes], height: int, width: int) -> np.ndarray:
        import cv2

        out = np.empty((len(blobs), height, width, 3), dtype=np.uint8)
        for idx, blob in enumerate(blobs):
            bgr = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Failed to decode video frame blob at position {idx}")
            out[idx] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return out


__all__ = ["LanceVideoFramesBackend", "reset_worker_state"]
