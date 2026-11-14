"""Check explicit neonatal compatibility handling and annotation boundaries."""

import importlib
import json

import numpy as np
import pytest
import torch
from scipy.io import savemat


pytest.importorskip("mne")
pyedflib = pytest.importorskip("pyedflib")
validation = importlib.import_module("profiling.neonatal_edf_validation")
experiment = importlib.import_module("profiling.edf_signal_loading")


def make_inconsistent_edf(path):
    """Model the observed source: EDF+C without TALs and one -32768 sample.

    Only fixture header bytes are modified after writing a valid ordinary EDF.
    The declared digital minimum becomes -32767 while samples remain unchanged.
    """
    from pyedflib import highlevel

    headers = [highlevel.make_signal_header(f"EEG{channel}", sample_frequency=128) for channel in range(2)]
    samples = np.arange(512, dtype=np.int32).reshape(2, 256)
    samples[0, 3] = -32768
    highlevel.write_edf(str(path), samples, headers, digital=True, file_type=pyedflib.FILETYPE_EDF)
    content = bytearray(path.read_bytes())
    content[192:236] = b"EDF+C".ljust(44)
    # EDF signal fields are grouped by field, then channel (16+80+8+8+8).
    first_digital_min = 256 + 2 * 120
    content[first_digital_min : first_digital_min + 8] = b"-32767  "
    path.write_bytes(content)


def test_normalization_preserves_every_other_byte_and_original(tmp_path):
    source, derivative = tmp_path / "eeg1.edf", tmp_path / "copy.edf"
    make_inconsistent_edf(source)
    original = source.read_bytes()
    result = validation.normalize_continuous_header(source, derivative)
    copied = derivative.read_bytes()
    assert source.read_bytes() == original
    assert copied[:192] == original[:192]
    assert copied[192:236] == b" " * 44
    assert copied[236:] == original[236:]
    assert result["all_other_bytes_identical"]
    with pytest.raises(FileExistsError):
        validation.normalize_continuous_header(source, derivative)


def test_annotation_bearing_edf_cannot_be_normalized(tmp_path):
    from pyedflib import highlevel

    source = tmp_path / "annotated.edf"
    headers = [highlevel.make_signal_header("EEG0", sample_frequency=128)]
    highlevel.write_edf(str(source), np.zeros((1, 256), dtype=np.int32), headers, digital=True)
    with pytest.raises(ValueError, match="annotation-bearing"):
        validation.normalize_continuous_header(source, tmp_path / "copy.edf")


def test_out_of_range_samples_require_opt_in_and_are_not_clipped(tmp_path):
    source, derivative = tmp_path / "eeg1.edf", tmp_path / "copy.edf"
    make_inconsistent_edf(source)
    validation.normalize_continuous_header(source, derivative)
    with pytest.raises(ValueError, match="declared digital bounds"):
        experiment.read_digital(derivative)
    recording = experiment.read_digital(derivative, allow_out_of_range_samples=True)
    assert recording.samples[3, 0] == -32768
    assert recording.metadata.headers[0]["digital_min"] == -32767


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_selected_windows_and_external_label_alignment(tmp_path, device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA GPU not available")
    source, derivative = tmp_path / "eeg1.edf", tmp_path / "copy.edf"
    make_inconsistent_edf(source)
    validation.normalize_continuous_header(source, derivative)
    labels = np.array([[0, 1], [0, 0], [1, 0]], dtype=np.int32)
    subjects = np.empty((1, 1), dtype=object)
    subjects[0, 0] = labels
    annotations = tmp_path / "annotations.mat"
    savemat(annotations, {"annotat_new": subjects})
    loaded = validation.load_expert_labels(annotations, 1)
    np.testing.assert_array_equal(loaded, labels)
    result = validation.validate_windows(
        source, derivative, loaded, torch.device(device_name), 128, allow_out_of_range_samples=True
    )
    assert result["out_of_range_samples_per_channel"] == [1, 0]
    assert result["expert_labels_shape"] == [3, 2]
    crossing = next(window for window in result["windows"] if window["start_sample"] == 127)
    assert crossing["annotation_seconds_half_open"] == [0, 2]
    assert crossing["expert_votes"] == labels.tolist()
    with pytest.raises(ValueError, match="coverage"):
        validation.validate_windows(
            source, derivative, labels[:, :1], torch.device(device_name), 128, allow_out_of_range_samples=True
        )


def test_full_record_window_stays_in_bounds():
    labels = np.array([[0, 1], [0, 0], [1, 0]], dtype=np.int32)
    assert validation.select_window_starts(256, 128, 256, labels) == [0]


def test_cache_preparation_requires_matching_sources_and_explicit_exceptions(tmp_path):
    """Connect EDF preparation to the generic cache without silently accepting repairs."""
    from profiling.prepare_signal_cache import prepare_cache
    from profiling.signal_cache import SignalCache

    source, derivative = tmp_path / "eeg1.edf", tmp_path / "copy.edf"
    make_inconsistent_edf(source)
    normalization = validation.normalize_continuous_header(source, derivative)
    report = {
        "checks": {"source_metadata_and_selected_windows": "passed"},
        "source": {"sha256": validation.file_digest(source)},
        "digital_source": {"sha256": validation.file_digest(derivative)},
        "normalization": normalization,
        "out_of_range_samples_per_channel": [1, 0],
        "out_of_range_samples_allowed": True,
    }
    report_path = tmp_path / "validation.json"
    report_path.write_text(json.dumps(report))
    cache_path = tmp_path / "cache"
    with pytest.raises(ValueError, match="exceptions"):
        prepare_cache(source, derivative, report_path, cache_path)
    assert not cache_path.exists()
    prepare_cache(source, derivative, report_path, cache_path, accept_source_exceptions=True)
    cache = SignalCache.open(cache_path)
    assert cache.digital[3, 0] == -32768
    assert cache.metadata.provenance["accepted_source_exceptions"]
    report["source"]["sha256"] = "wrong digest"
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="checksum"):
        prepare_cache(source, derivative, report_path, tmp_path / "invalid", accept_source_exceptions=True)
