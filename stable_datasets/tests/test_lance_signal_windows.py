"""Real Lance range/chunk reads and the current-cache baseline share one contract."""

import hashlib
import json
import pickle

import numpy as np
import pytest


pytest.importorskip("lance")

from profiling.current_signal_cache import CurrentCacheWindowReader, CurrentSignalCache, write_current_cache
from profiling.lance_signal_windows import LanceSignalWindows, write_signal_store
from profiling.signal_windows import Recording, Window, physical_windows


class CountingRows:
    def __init__(self, dataset):
        self.dataset = dataset
        self.calls = []

    def __getitem__(self, index):
        self.calls.append(index)
        return self.dataset[index]


@pytest.fixture
def signals():
    arrays = [np.arange(22, dtype=np.int16).reshape(11, 2), np.arange(21, dtype=np.int16).reshape(7, 3) - 12]
    metas = [
        Recording(
            len(a),
            2.0,
            tuple(f"ch{i}" for i in range(a.shape[1])),
            tuple(1e-6 * (i + 1) for i in range(a.shape[1])),
            tuple(-2e-6 for _ in range(a.shape[1])),
        )
        for a in arrays
    ]
    return metas, arrays


@pytest.mark.parametrize("chunk_samples", [0, 1, 4, 20])
def test_real_layouts_preserve_cross_chunk_windows_and_duplicates(tmp_path, signals, chunk_samples):
    metas, arrays = signals
    path = tmp_path / "signals.lance"
    write_signal_store(path, metas, arrays, chunk_samples=chunk_samples)
    store = LanceSignalWindows(path)
    requests = [Window(1, 3, 4), Window(0, 0, 11), Window(0, 3, 5), Window(1, 3, 4)]
    for actual, r in zip(store.read(requests), requests):
        np.testing.assert_array_equal(actual, arrays[r.recording][r.start : r.start + r.length])
    assert store.recordings == metas
    assert store.read([]) == []
    with pytest.raises(FileExistsError):
        write_signal_store(path, metas, arrays, chunk_samples=chunk_samples)


def test_blob_uses_exact_byte_ranges_without_full_row_materialization(tmp_path, signals):
    metas, arrays = signals
    write_signal_store(tmp_path / "blob", metas, arrays)
    store = LanceSignalWindows(tmp_path / "blob")
    real = store.dataset
    calls = []

    class RangeOnly:
        def read_blob_ranges(self, column, ranges, **kwargs):
            calls.append((column, ranges, kwargs))
            return real.read_blob_ranges(column, ranges, **kwargs)

        def take(self, *args, **kwargs):
            pytest.fail("Recording blob must not be loaded by whole-row take")

    store.dataset = RangeOnly()
    requests = [Window(1, 2, 3), Window(0, 1, 2), Window(1, 2, 3)]
    store.read(requests)
    assert calls[0][1] == [(1, 12, 18), (0, 4, 8), (1, 12, 18)]
    assert store.payload_bytes(requests) == 44


def test_chunk_payload_counts_unique_rows_and_truncated_last_chunk(tmp_path, signals):
    metas, arrays = signals
    write_signal_store(tmp_path / "chunks", metas, arrays, chunk_samples=4)
    store = LanceSignalWindows(tmp_path / "chunks")
    requests = [Window(0, 7, 4), Window(0, 8, 3), Window(1, 6, 1)]
    assert store.payload_bytes(requests) == (4 + 3) * 2 * 2 + 3 * 3 * 2
    for requests in [[Window(-1, 0, 1)], [Window(0, -1, 1)], [Window(1, 6, 2)], [Window(0, 0, 0)]]:
        with pytest.raises(ValueError):
            store.read(requests)


def test_production_baseline_materializes_whole_recordings_but_matches_windows(tmp_path, signals):
    metas, arrays = signals
    whole = [Window(i, 0, m.n_samples) for i, m in enumerate(metas)]
    physical = physical_windows(metas, whole, arrays)
    write_current_cache(tmp_path / "current", physical)
    baseline = CurrentSignalCache(tmp_path / "current", metas)
    rows = CountingRows(baseline.dataset)
    baseline.dataset = rows
    requests = [Window(0, 3, 2), Window(1, 4, 3), Window(0, 3, 2)]
    expected = physical_windows(metas, requests, [arrays[r.recording][r.start : r.start + r.length] for r in requests])
    for actual, reference in zip(baseline.read(requests), expected):
        np.testing.assert_array_equal(actual, reference)
    assert rows.calls == [0, 1]
    assert baseline.materialized_samples(requests) == 11 * 2 + 7 * 3


@pytest.fixture
def prepared_edf(tmp_path):
    pytest.importorskip("mne")
    pytest.importorskip("edfio")
    pytest.importorskip("pyedflib")
    from pyedflib import highlevel

    from profiling.benchmark_signal_windows import prepare

    source = tmp_path / "known.edf"
    values = np.stack([np.arange(2048) - 1000, np.arange(2048) // 2 - 700]).astype(np.int32)
    headers = [
        highlevel.make_signal_header(f"EEG{i}", sample_frequency=128, physical_min=-100, physical_max=300)
        for i in range(2)
    ]
    header = highlevel.make_header()
    header["annotations"] = [[1.0, 0.5, "event"]]
    highlevel.write_edf(str(source), values, headers, header, digital=True)
    prepare(source, tmp_path / "workspace", chunk_samples=200)
    return source, tmp_path / "workspace", values


def test_edf_preparation_and_matched_cpu_benchmark(tmp_path, prepared_edf):
    from profiling.benchmark_signal_windows import plot, prepare, run

    source, workspace, values = prepared_edf
    original = source.read_bytes()
    prepared = json.loads((tmp_path / "workspace/preparation.json").read_text())
    assert prepared["annotations"] == [{"onset_s": 1.0, "duration_s": 0.5, "text": "event"}]
    report = run(
        source,
        tmp_path / "workspace",
        batch_sizes=[1, 4],
        seconds=4,
        device="cpu",
        repeats=1,
        storage_label="test-local",
    )
    assert report["samples_per_window"] == 512
    assert report["endpoint"] == "float32 NumPy batch in volts"
    for row in report["reports"]:
        assert set(row["results"]) == {"current_cache", "edf_memmap", "lance_blob_ranges", "lance_chunk_rows"}
        assert row["current_materialized_scalars"] == values.size
        assert max(row["max_abs_errors_volts"].values()) < 1e-10
        assert all(v["windows_per_second"] > 0 for v in row["results"].values())
    assert source.read_bytes() == original
    result = tmp_path / "result.json"
    result.write_text(json.dumps(report))
    plot(result, tmp_path / "plot.png")
    assert (tmp_path / "plot.png").stat().st_size > 0
    with pytest.raises(FileExistsError):
        prepare(source, tmp_path / "workspace", chunk_samples=200)
    source.write_bytes(original + b"changed")
    with pytest.raises(ValueError, match="differs"):
        CurrentCacheWindowReader(source, workspace)
    with pytest.raises(ValueError, match="differs"):
        run(source, tmp_path / "workspace", batch_sizes=[1], seconds=4, device="cpu", repeats=1, storage_label="test")


def test_prepared_cache_windows_preserve_values_grouping_and_reopen(prepared_edf):
    from profiling.edf_record_loading import EdfWindowReader

    source, workspace, _ = prepared_edf
    reader = CurrentCacheWindowReader(source, workspace)
    rows = CountingRows(reader._cache.dataset)
    reader._cache.dataset = rows
    starts = np.array([1792, 0, 127, 1792])
    expected = EdfWindowReader(source).read(starts, 256)
    np.testing.assert_array_equal(reader.read(starts, 256).numpy(), expected.numpy())
    assert rows.calls == [0]
    np.testing.assert_array_equal(reader.read(starts, 256).numpy(), expected.numpy())
    assert rows.calls == [0, 0]
    assert reader.provenance["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert (
        reader.provenance["preparation_sha256"]
        == hashlib.sha256((workspace / "preparation.json").read_bytes()).hexdigest()
    )
    restored = pickle.loads(pickle.dumps(reader))
    assert restored._cache is None and restored._pid is None
    np.testing.assert_array_equal(restored.read(starts, 256).numpy(), expected.numpy())
    assert restored._cache is not reader._cache


@pytest.mark.parametrize("field", ["source_sha256", "channels", "rate_hz", "n_samples", "gain", "offset"])
def test_prepared_cache_rejects_wrong_source_and_recording_metadata(prepared_edf, field):
    source, workspace, _ = prepared_edf
    manifest = workspace / "preparation.json"
    prepared = json.loads(manifest.read_text())
    if field == "source_sha256":
        prepared[field] = "0" * 64
    elif field == "channels":
        prepared["recording"][field].reverse()
    elif field in ("gain", "offset"):
        prepared["recording"][field][0] += 1
    else:
        prepared["recording"][field] += 1
    manifest.write_text(json.dumps(prepared))
    with pytest.raises(ValueError, match="source|metadata"):
        CurrentCacheWindowReader(source, workspace)


@pytest.mark.parametrize(
    "starts,length",
    [(np.array([-1]), 256), (np.array([1793]), 256), (np.array([]), 256), (np.array([0.5]), 256), (np.array([0]), 0)],
)
def test_prepared_cache_rejects_invalid_windows_before_materialization(prepared_edf, starts, length):
    source, workspace, _ = prepared_edf
    reader = CurrentCacheWindowReader(source, workspace)
    rows = CountingRows(reader._cache.dataset)
    reader._cache.dataset = rows
    with pytest.raises(ValueError):
        reader.read(starts, length)
    assert rows.calls == []
