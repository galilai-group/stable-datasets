"""Production float32 nested-sequence cache baseline for the window study.

The experiment uses the production writer, backend and formatter for the
neonatal signal column. It excludes annotation fields, labels and downloads;
it does not claim to run the full SeizuresNeonatal builder. Each recording is
materialized once per batch, avoiding a deliberately weak repeated-row baseline.
"""

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
