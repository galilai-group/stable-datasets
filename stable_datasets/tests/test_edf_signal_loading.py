"""Optional EDF experiment checks; requires MNE and pyEDFlib."""

import importlib
import json
from dataclasses import replace

import numpy as np
import pytest
import torch


mne = pytest.importorskip("mne")
pyedflib = pytest.importorskip("pyedflib")
experiment = importlib.import_module("profiling.edf_signal_loading")


def make_edf(path, rates=(128, 128), dimension="uV"):
    """Write known integer samples with nonzero calibration offset."""
    from pyedflib import highlevel

    headers = [
        highlevel.make_signal_header(
            f"EEG{i}", sample_frequency=rate, dimension=dimension, physical_min=-100, physical_max=300
        )
        for i, rate in enumerate(rates)
    ]
    samples = [np.arange(rate * 2, dtype=np.int32) - 100 for rate in rates]
    header = highlevel.make_header()
    header["annotations"] = [[0.5, 0.25, "test-event"]]
    assert highlevel.write_edf(str(path), samples, headers, header, digital=True)
    return samples


def test_real_edf_integer_calibration_units_order_and_annotation(tmp_path):
    path = tmp_path / "known.edf"
    samples = make_edf(path)
    recording = experiment.read_digital(path)
    np.testing.assert_array_equal(recording.samples, np.stack(samples, axis=1))
    assert recording.samples.dtype == np.int16
    assert [header["label"] for header in recording.metadata.headers] == ["EEG0", "EEG1"]
    assert recording.metadata.sample_rate_hz == 128
    assert recording.metadata.annotation_onset_s == [0.5]
    assert recording.metadata.annotation_duration_s == [0.25]
    assert recording.metadata.annotation_description == ["test-event"]
    np.testing.assert_allclose(
        recording.samples * recording.gain + recording.offset, experiment.read_mne(path), rtol=2e-6, atol=1e-10
    )


@pytest.mark.parametrize(
    "rates,dimension,message",
    [((128, 64), "uV", "Mixed-rate"), ((128, 128), "Ohm", "voltage")],
)
def test_unsupported_rates_and_units_are_rejected(tmp_path, rates, dimension, message):
    path = tmp_path / "unsupported.edf"
    make_edf(path, rates, dimension)
    with pytest.raises(ValueError, match=message):
        experiment.read_digital(path)


def test_discontinuous_edf_is_rejected_before_reading(tmp_path):
    path = tmp_path / "discontinuous.edf"
    header = bytearray(b" " * 256)
    header[:8] = b"0       "
    header[192:197] = b"EDF+D"
    path.write_bytes(header)
    with pytest.raises(ValueError, match="continuous"):
        experiment.read_digital(path)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize("window_length", [128, 256])
def test_experiment_paths_and_report_contract(tmp_path, device_name, window_length):
    """Exercise all device stages and serialization, including full-length windows."""
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA GPU not available")
    source_path = tmp_path / "known.edf"
    make_edf(source_path)
    config = experiment.ExperimentConfig(
        edf_path=source_path,
        device=torch.device(device_name),
        batch_size=3,
        window_length=window_length,
        repeats=1,
        warmup=0,
        output_path=tmp_path / "results.json",
    )
    prepared = experiment.prepare_experiment(config)
    measurements, checks = experiment.run_benchmarks(config, prepared)
    report = experiment.build_report(config, prepared, measurements, checks)
    experiment.write_report(config.output_path, report)
    saved_report = json.loads(config.output_path.read_text())

    assert saved_report["shape"] == [3, window_length, 2]
    assert saved_report["metadata"]["annotation_onset_s"] == [0.5]
    assert saved_report["checks"]["channel_order_rates_units_annotations"] == "passed"
    assert saved_report["checks"]["cpu_full_record_max_abs_error_volts"] < 1e-10
    expected_stages = {
        "preparation_mne_full_record_float32",
        "preparation_pyedflib_full_record_int16",
        "cached_numpy_float32_windows",
        "cached_arrow_float32_windows",
        "batch_numpy_cpu_calibration",
        "batch_torch_cpu_calibration",
    }
    if device_name == "cuda":
        expected_stages.update(
            {
                "batch_numpy_calibrate_pin_transfer",
                "batch_torch_calibrate_pin_transfer",
                "batch_int16_pin_transfer_gpu_calibrate",
                "cached_numpy_float32_gather_pin_transfer",
                "cached_int16_gather_pin_transfer_gpu_calibrate",
            }
        )
        assert saved_report["checks"]["cuda_full_record_max_abs_error_volts"] < 1e-10
    assert set(saved_report["stages"]) == expected_stages
    assert all(set(result) == {"median_ms", "p95_ms"} for result in measurements.values())
    # A repeat write must preserve the first report, even after CLI validation.
    original_output = config.output_path.read_bytes()
    with pytest.raises(FileExistsError):
        experiment.write_report(config.output_path, report)
    assert config.output_path.read_bytes() == original_output


def test_reference_metadata_mismatch_is_rejected(tmp_path):
    """Mandatory metadata checks must raise explicitly, not rely on assert."""
    path = tmp_path / "known.edf"
    make_edf(path)
    metadata = experiment.read_digital(path).metadata
    with mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR") as reference:
        with pytest.raises(ValueError, match="sampling rate"):
            experiment.validate_reference_metadata(reference, replace(metadata, sample_rate_hz=64))
