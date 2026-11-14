"""Regression checks for the shared helpers' dependency and timing contracts."""

import hashlib
import subprocess
import sys

import pytest
import torch

from profiling import signal_utils


@pytest.mark.parametrize("payload", [b"", b"signal" * 200_000])
def test_checksum_handles_empty_and_multiblock_files(tmp_path, payload):
    source = tmp_path / "samples.bin"
    source.write_bytes(payload)
    assert signal_utils.checksum(source) == hashlib.sha256(payload).hexdigest()


def test_cuda_stage_waits_for_completed_work_and_excludes_warmup(monkeypatch):
    events = []
    timestamps = iter([1.0, 1.002, 2.0, 2.004])
    monkeypatch.setattr(signal_utils.time, "perf_counter", lambda: next(timestamps))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("sync"))
    result = signal_utils.measure_stage(lambda: events.append("call"), torch.device("cuda"), 1, 2)
    assert events == ["call", "sync", "sync", "call", "sync", "sync", "call", "sync"]
    assert result == pytest.approx({"median_ms": 3.0, "p95_ms": 3.9})


def test_native_benchmarks_import_without_legacy_experiments():
    for reader in ("mne", "edfio", "wfdb", "numba", "triton"):
        pytest.importorskip(reader)
    # A fresh interpreter detects transitive dependencies even when other tests
    # have already imported the legacy modules in the main pytest process.
    script = """
import importlib.abc
import sys
import torch

class RejectLegacyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {
            'profiling.signal_loading', 'profiling.signal_cache',
            'profiling.edf_signal_loading', 'pyarrow', 'pyedflib',
        }:
            raise ImportError(f'Unexpected legacy dependency: {fullname}')

sys.meta_path.insert(0, RejectLegacyImports())
import profiling.benchmark_edf_records
import profiling.benchmark_wfdb212
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
