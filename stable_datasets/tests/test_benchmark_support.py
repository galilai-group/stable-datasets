"""Regression checks for the shared helpers' dependency and timing contracts."""

import hashlib
import subprocess
import sys

import pytest
import torch

from profiling import benchmark_support


@pytest.mark.parametrize("payload", [b"", b"signal" * 200_000])
def test_checksum_handles_empty_and_multiblock_files(tmp_path, payload):
    source = tmp_path / "samples.bin"
    source.write_bytes(payload)
    assert benchmark_support.checksum(source) == hashlib.sha256(payload).hexdigest()


def test_cuda_stage_waits_for_completed_work_and_excludes_warmup(monkeypatch):
    events = []
    timestamps = iter([1.0, 1.002, 2.0, 2.004])
    monkeypatch.setattr(benchmark_support.time, "perf_counter", lambda: next(timestamps))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("sync"))
    result = benchmark_support.measure_stage(lambda: events.append("call"), torch.device("cuda"), 1, 2)
    assert events == ["call", "sync", "sync", "call", "sync", "sync", "call", "sync"]
    assert result == pytest.approx({"median_ms": 3.0, "p95_ms": 3.9})


@pytest.mark.parametrize(
    "module,dependency",
    [("profiling.edf_source", "edfio")],
)
def test_sources_and_cpu_decoder_import_without_benchmark_or_gpu_dependencies(module, dependency):
    pytest.importorskip(dependency)
    script = """
import importlib.abc
import sys

class RejectUnrelatedImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mne', 'torch', 'triton', 'pyarrow', 'pyedflib'}:
            raise ImportError(f'Unexpected source-reader dependency: {fullname}')
        if fullname.startswith('profiling.benchmark') or fullname == 'profiling.edf_reference':
            raise ImportError(f'Unexpected benchmark dependency: {fullname}')

sys.meta_path.insert(0, RejectUnrelatedImports())
"""
    script += f"\nimport {module}\n"
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
