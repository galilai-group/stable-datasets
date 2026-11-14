"""Native original-source EDF checks, independent of processed caches."""

import importlib

import numpy as np
import pytest
import torch


mne = pytest.importorskip("mne")
edfio = pytest.importorskip("edfio")
pyedflib = pytest.importorskip("pyedflib")
native = importlib.import_module("profiling.native_edf_loading")


def make_source(path, *, mixed_rates=False):
    """Write known affine calibration, an annotation and integer samples."""
    from pyedflib import highlevel

    rates = [128, 64 if mixed_rates else 128]
    headers = [
        highlevel.make_signal_header(f"EEG{index}", sample_frequency=rate, physical_min=-100, physical_max=300)
        for index, rate in enumerate(rates)
    ]
    data = [np.arange(rate * 3, dtype=np.int32) - 100 for rate in rates]
    header = highlevel.make_header()
    header["annotations"] = [[0.5, 0.25, "test-event"]]
    highlevel.write_edf(str(path), data, headers, header, digital=True)
    return data


def test_native_windows_preserve_order_duplicates_and_cross_record_boundaries(tmp_path):
    path = tmp_path / "source.edf"
    data = make_source(path)
    reader = native.NativeEdfReader(path)
    starts = np.array([127, 0, 127, 256])
    actual = reader.read_windows(starts, 128)
    expected = np.stack([np.stack(data, axis=1)[start : start + 128] for start in starts]).astype(np.int16)
    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.c_contiguous
    with mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR") as reference:
        assert (
            native.validate_source(reader, reference, starts, 128)["channel_order_clock_start_annotations"] == "passed"
        )
    with pytest.raises(ValueError, match="outside"):
        reader.read_windows(np.array([257]), 128)


def test_native_reader_rejects_unsupported_rates_and_encoding(tmp_path):
    path = tmp_path / "mixed.edf"
    make_source(path, mixed_rates=True)
    with pytest.raises(ValueError, match="Mixed sample rates"):
        native.NativeEdfReader(path)
    invalid = tmp_path / "unsupported.edf"
    invalid.write_bytes(b"\xffBIOSEMI")
    with pytest.raises(ValueError, match="16-bit EDF"):
        native.NativeEdfReader(invalid)


def test_inconsistent_original_is_read_without_header_or_sample_rewrite(tmp_path):
    from pyedflib import highlevel

    path = tmp_path / "original.edf"
    headers = [highlevel.make_signal_header("EEG0", sample_frequency=128)]
    samples = np.arange(384, dtype=np.int32)[None]
    samples[0, 3] = -32768
    highlevel.write_edf(str(path), samples, headers, digital=True, file_type=pyedflib.FILETYPE_EDF)
    content = bytearray(path.read_bytes())
    content[192:236] = b"EDF+C".ljust(44)
    content[376:384] = b"-32767  "  # One channel: digital-min field begins at 256+120.
    path.write_bytes(content)
    before = path.read_bytes()
    report = native.run_benchmark(path, torch.device("cpu"), 3, 128, 1)
    assert report["source_unchanged"]
    assert report["checks"]["selected_samples_outside_declared_bounds"] >= 1
    assert path.read_bytes() == before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available")
def test_every_original_source_gpu_path_matches_mne(tmp_path):
    path = tmp_path / "source.edf"
    make_source(path)
    report = native.run_benchmark(path, torch.device("cuda"), 3, 128, 1)
    assert report["source_unchanged"]
    assert len(report["stages"]) == 11
