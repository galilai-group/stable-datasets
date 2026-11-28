"""CPU-only, process-local batch readers for the EEG training diagnostic.

One dataset item is an entire predetermined batch. This preserves the window
benchmark's grouping of recording-row requests. Workers never initialize CUDA;
the existing GPU assembler remains in the training process with zero workers.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from profiling.signal_windows import Recording, Window, physical_windows, validate_windows


CPU_PATHS = ("current_cache", "edf_memmap", "lance_blob_ranges", "lance_chunk_rows")
GPU_PATH = "edf_gpu_assembly"


def configure_reader_threads(worker_id: int = 0) -> None:
    """Limit each reader process to one CPU thread, including Arrow pools."""
    import pyarrow

    del worker_id
    torch.set_num_threads(1)
    pyarrow.set_cpu_count(1)
    pyarrow.set_io_thread_count(1)


class WindowBatches(Dataset):
    """Open storage lazily in each worker and return matched CPU batches."""

    def __init__(self, source: Path, workspace: Path, starts: np.ndarray, targets: np.ndarray, length: int, path: str):
        if path not in (*CPU_PATHS, GPU_PATH):
            raise ValueError("Unknown loading path")
        if (
            starts.ndim != 2
            or not all(starts.shape)
            or targets.shape != starts.shape
            or not np.issubdtype(starts.dtype, np.integer)
            or not np.isin(targets, [0, 1]).all()
        ):
            raise ValueError("Require nonempty matching batch schedules and binary targets")
        self.source, self.workspace, self.path = Path(source), Path(workspace), path
        self.metadata = Recording.from_dict(json.loads((self.workspace / "preparation.json").read_text())["recording"])
        validate_windows([self.metadata], [Window(0, int(start), length) for start in starts.flat])
        self.starts = starts.astype(np.int64, copy=True)
        self.targets = targets.astype(np.int64, copy=True)
        self.length = length
        self._reader = None
        self._pid = None

    def __len__(self):
        return len(self.starts)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_reader=None, _pid=None)
        return state

    def _open(self):
        if self._reader is not None and self._pid == os.getpid():
            return
        if self.path == "current_cache":
            from profiling.current_signal_cache import CurrentSignalCache

            self._reader = CurrentSignalCache(self.workspace / "current", [self.metadata])
        elif self.path == "edf_memmap":
            from profiling.edf_record_loading import BatchedEdfRecords
            from profiling.edf_source import NativeEdfReader

            self._reader = BatchedEdfRecords(self.source, NativeEdfReader(self.source))
        else:
            from profiling.lance_signal_windows import LanceSignalWindows

            name = "recordings.lance" if self.path == "lance_blob_ranges" else "chunks.lance"
            self._reader = LanceSignalWindows(self.workspace / name)
            if self._reader.recordings != [self.metadata]:
                raise ValueError("Lance recording metadata disagree")
        self._pid = os.getpid()

    def __getitem__(self, index):
        starts = self.starts[index]
        batch = {"starts": starts.copy(), "targets": self.targets[index].copy()}
        if self.path == GPU_PATH:
            return batch
        self._open()
        requests = [Window(0, int(start), self.length) for start in starts]
        if self.path == "edf_memmap":
            from profiling.signal_calibration import calibrate_cpu

            raw = self._reader.read_cpu_from_records(self._reader.plan(starts, self.length))
            batch["values"] = calibrate_cpu(raw, self._reader.metadata.gain, self._reader.metadata.offset)
        elif self.path == "current_cache":
            batch["values"] = np.stack(self._reader.read(requests))
        else:
            batch["values"] = np.stack(physical_windows([self.metadata], requests, self._reader.read(requests)))
        return batch
