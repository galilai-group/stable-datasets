"""Production float32 nested-sequence cache baseline for the window study.

The experiment uses the production writer, backend and formatter for the
neonatal signal column. It excludes annotation fields, labels and downloads;
it does not claim to run the full SeizuresNeonatal builder. Each recording is
materialized once per batch, avoiding a deliberately weak repeated-row baseline.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from profiling.signal_windows import Recording, Window, validate_windows


def _features():
    from stable_datasets.schema import Features, Sequence, Value

    return Features({"series": Sequence(Sequence(Value("float32")))})


def write_current_cache(path: Path, physical: list[np.ndarray]) -> None:
    """Write one whole recording per row with production Arrow-cache defaults."""
    from stable_datasets.cache import write_sharded_arrow_cache

    if path.exists():
        raise FileExistsError(path)
    write_sharded_arrow_cache(((i, {"series": values}) for i, values in enumerate(physical)), _features(), path)


class CurrentSignalCache:
    """Read through StableDataset, then select windows from full Python rows."""

    def __init__(self, path: Path, recordings: list[Recording]):
        from stable_datasets.cache import open_cache
        from stable_datasets.dataset import StableDataset
        from stable_datasets.schema import DatasetInfo

        self.recordings = recordings
        features = _features()
        opened = open_cache(path, features)
        self.dataset = StableDataset(
            features, DatasetInfo(features=features), backend=opened.backend, num_rows=opened.num_rows
        )
        if len(self.dataset) != len(recordings):
            raise ValueError("Recording index and baseline cache disagree")

    def read(self, requests: list[Window]) -> list[np.ndarray]:
        """Return float32 volts; grouping prevents repeated recording reads per batch."""
        validate_windows(self.recordings, requests)
        rows = {i: self.dataset[i]["series"] for i in sorted({r.recording for r in requests})}
        return [np.asarray(rows[r.recording][r.start : r.start + r.length], dtype=np.float32) for r in requests]

    def materialized_samples(self, requests: list[Window]) -> int:
        """Count logical scalar values materialized; this is not physical disk I/O."""
        validate_windows(self.recordings, requests)
        return sum(
            self.recordings[i].n_samples * len(self.recordings[i].channels) for i in {r.recording for r in requests}
        )


class CurrentCacheWindowReader:
    """Expose source-verified prepared cache windows with process-local handles."""

    loading_path = "current_cache"

    def __init__(self, source: Path, workspace: Path):
        from profiling.benchmark_support import checksum
        from profiling.edf_source import NativeEdfReader

        self.workspace = Path(workspace).expanduser().resolve()
        preparation = (self.workspace / "preparation.json").read_bytes()
        prepared = json.loads(preparation)
        digest = checksum(Path(source))
        if prepared["source_sha256"] != digest:
            raise ValueError("EDF differs from the prepared cache source")
        self.metadata = NativeEdfReader(Path(source)).metadata
        m = self.metadata
        self.recording = Recording.from_dict(prepared["recording"])
        expected = Recording(m.sample_count, m.sample_rate_hz, m.channel_names, tuple(m.gain), tuple(m.offset))
        if self.recording != expected:
            raise ValueError("Prepared cache recording metadata disagree with EDF")
        self.provenance = {
            "workspace": str(self.workspace),
            "preparation_sha256": hashlib.sha256(preparation).hexdigest(),
            "source_sha256": digest,
        }
        self._cache = None
        self._pid = None
        self._open()

    def _open(self):
        if self._cache is None or self._pid != os.getpid():
            self._cache = CurrentSignalCache(self.workspace / "current", [self.recording])
            self._pid = os.getpid()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_cache=None, _pid=None)
        return state

    def read(self, starts: np.ndarray, length: int):
        """Return CPU float32 volts, materializing the recording once per batch."""
        import torch

        if starts.ndim != 1 or not np.issubdtype(starts.dtype, np.integer) or not len(starts):
            raise ValueError("Require a nonempty integer start vector")
        requests = [Window(0, int(start), length) for start in starts]
        validate_windows([self.recording], requests)
        self._open()
        values = np.stack(self._cache.read(requests))
        if values.shape != (len(starts), length, len(self.recording.channels)):
            raise ValueError("Prepared cache window shape disagrees with EDF")
        return torch.from_numpy(values)
