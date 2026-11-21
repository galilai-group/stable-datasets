"""Two experimental Lance int16 layouts; no calibration or training policy.

Recording blobs use Lance's planned byte-range API. Chunk rows use ordinary
binary cells and one deduplicated take per batch. Payload counters describe
requested logical bytes, not filesystem/page-cache or physical disk traffic.
"""

import json
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa

from profiling.signal_windows import Recording, Window, validate_windows


_KEY = b"stable_datasets.research.signal_windows"


def write_signal_store(path: Path, recordings: list[Recording], samples: list[np.ndarray], *, chunk_samples: int = 0):
    """Create an immutable experiment store. Zero chunk size selects recording blobs.

    Each recording is time-major little-endian int16. Metadata stores sample
    counts, clocks, channel order, calibration and the first dataset row index.
    Dataset row indices are stable only for this immutable store version; no
    compaction/append protocol or production signal feature is introduced.
    """
    if Path(path).exists():
        raise FileExistsError(path)
    if not recordings or len(recordings) != len(samples) or chunk_samples < 0 or not isinstance(chunk_samples, int):
        raise ValueError("Require matched nonempty recording vectors and an integer chunk size >= 0")
    offsets, row = [], 0
    for meta, values in zip(recordings, samples):
        if values.dtype != np.int16 or values.shape != (meta.n_samples, len(meta.channels)):
            raise ValueError("Source samples must match recording metadata and be int16")
        offsets.append(row)
        row += (meta.n_samples + chunk_samples - 1) // chunk_samples if chunk_samples else 1
    info = {
        "version": 1,
        "chunk_samples": chunk_samples,
        "row_offsets": offsets,
        "recordings": [m.to_dict() for m in recordings],
    }
    field = pa.field(
        "samples", pa.large_binary(), metadata=None if chunk_samples else {b"lance-encoding:blob": b"true"}
    )
    schema = pa.schema([field], metadata={_KEY: json.dumps(info).encode()})

    def batches():
        for meta, values in zip(recordings, samples):
            stride = chunk_samples or meta.n_samples
            for start in range(0, meta.n_samples, stride):
                payload = np.asarray(values[start : start + stride], dtype="<i2").tobytes(order="C")
                yield pa.RecordBatch.from_arrays([pa.array([payload], type=pa.large_binary())], schema=schema)

    reader = pa.RecordBatchReader.from_batches(schema, batches())
    lance.write_dataset(reader, str(path), data_storage_version="2.2", mode="create")


class LanceSignalWindows:
    """Read the same sample-indexed window contract from either immutable layout."""

    def __init__(self, path: Path):
        self.dataset = lance.dataset(str(path))
        info = json.loads(self.dataset.schema.metadata[_KEY])
        if info["version"] != 1:
            raise ValueError("Unsupported experiment store version")
        self.recordings = [Recording.from_dict(x) for x in info["recordings"]]
        self.chunk_samples = info["chunk_samples"]
        self.row_offsets = info["row_offsets"]
        if not self.chunk_samples and not hasattr(self.dataset, "read_blob_ranges"):
            raise RuntimeError("Recording-blob comparison requires Lance.read_blob_ranges; no full-blob fallback")

    def read(self, requests: list[Window]) -> list[np.ndarray]:
        """Return ordered int16 windows, including repeated and cross-chunk requests."""
        validate_windows(self.recordings, requests)
        if not requests:
            return []
        if self.chunk_samples:
            return self._read_chunks(requests)
        ranges = []
        for request in requests:
            width = len(self.recordings[request.recording].channels) * 2
            ranges.append((self.row_offsets[request.recording], request.start * width, request.length * width))
        output = [None] * len(requests)
        results = self.dataset.read_blob_ranges("samples", ranges, selector="indices", preserve_order=True)
        for i, _, payload in results:
            request = requests[i]
            shape = (request.length, len(self.recordings[request.recording].channels))
            if payload is None or len(payload) != shape[0] * shape[1] * 2 or output[i] is not None:
                raise ValueError("Blob range reader returned an invalid payload")
            output[i] = np.frombuffer(payload, dtype="<i2").reshape(shape)
        if any(x is None for x in output):
            raise ValueError("Blob range reader omitted a request")
        return output

    def _chunk_rows(self, requests):
        return sorted(
            {
                self.row_offsets[r.recording] + i
                for r in requests
                for i in range(r.start // self.chunk_samples, (r.start + r.length - 1) // self.chunk_samples + 1)
            }
        )

    def _read_chunks(self, requests):
        rows = self._chunk_rows(requests)
        payloads = self.dataset.take(rows, columns=["samples"])["samples"].to_pylist()
        blocks = dict(zip(rows, payloads))
        output = []
        for request in requests:
            channels = len(self.recordings[request.recording].channels)
            first = request.start // self.chunk_samples
            last = (request.start + request.length - 1) // self.chunk_samples
            parts = [
                np.frombuffer(blocks[self.row_offsets[request.recording] + i], dtype="<i2").reshape(-1, channels)
                for i in range(first, last + 1)
            ]
            values = np.concatenate(parts)
            start = request.start % self.chunk_samples
            output.append(values[start : start + request.length])
        return output

    def payload_bytes(self, requests: list[Window]) -> int:
        """Return selected blob-range bytes or unique full-chunk bytes, excluding metadata."""
        validate_windows(self.recordings, requests)
        if not self.chunk_samples:
            return sum(r.length * len(self.recordings[r.recording].channels) * 2 for r in requests)
        rows = set(self._chunk_rows(requests))
        size = 0
        for recording, meta in enumerate(self.recordings):
            for row in rows:
                chunk = row - self.row_offsets[recording]
                if 0 <= chunk < (meta.n_samples + self.chunk_samples - 1) // self.chunk_samples:
                    size += (
                        min(self.chunk_samples, meta.n_samples - chunk * self.chunk_samples) * len(meta.channels) * 2
                    )
        return size
