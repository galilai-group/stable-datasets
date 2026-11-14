"""Compare real EDF signal loading and calibration against MNE.

This isolated experiment accepts continuous EDF/EDF+C with equal-rate voltage
channels. pyEDFlib reads native samples on CPU; MNE supplies the float64 reference
in volts. Preparation and repeated in-memory batches are measured separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pyarrow as pa
import pyedflib
import torch
from numpy.typing import NDArray

from profiling.signal_loading import calibrate, calibration_parameters, extract_windows, measure_stage


# The absolute tolerance is 0.1 nV, below the reference recording's 1 uV step.
SIGNAL_RELATIVE_TOLERANCE = 2e-6
SIGNAL_ABSOLUTE_TOLERANCE_VOLTS = 1e-10
ANNOTATION_TOLERANCE_SECONDS = 1e-7
VOLTAGE_UNIT_SCALES = {"V": 1.0, "mV": 1e-3, "uV": 1e-6, "µV": 1e-6}
EDF_VERSION_SLICE = slice(0, 8)
EDF_RESERVED_SLICE = slice(192, 236)

Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]
BenchmarkOperations = dict[str, Callable[[], object]]
StageMeasurements = dict[str, dict[str, float]]


@dataclass(frozen=True)
class ExperimentConfig:
    """CLI settings; window length counts samples, not seconds."""

    edf_path: Path
    device: torch.device
    batch_size: int
    window_length: int
    repeats: int
    warmup: int
    output_path: Path


@dataclass(frozen=True)
class RecordingMetadata:
    """Channel order, original units and annotation times in seconds.

    Header dictionaries retain all pyEDFlib fields for JSON provenance. Physical
    bounds remain in source units; calibrated arrays use volts.
    """

    headers: list[dict[str, str | float | int]]
    sample_rate_hz: float
    sample_count: int
    start_datetime: str
    annotation_onset_s: list[float]
    annotation_duration_s: list[float]
    annotation_description: list[str]


@dataclass(frozen=True)
class DigitalRecording:
    """Int16 (time, channels) samples and float32 per-channel calibration.

    Multiplying samples by gain and adding offset yields volts. Arrays remain
    mutable; frozen fields prevent accidental reassignment only.
    """

    samples: NDArray[np.int16]
    gain: Float32Array
    offset: Float32Array
    metadata: RecordingMetadata


@dataclass(frozen=True)
class PreparedExperiment:
    """Inputs allocated outside timed batch operations.

    Recording arrays have shape (time, channels). Indices select identical
    (batch, length, channels) windows for every path. Batch-only stages reuse
    gathered integer samples; cached stages gather per call.
    """

    recording: DigitalRecording
    reference_volts: Float64Array
    cached_float32: Float32Array
    window_starts: NDArray[np.int64]
    window_indices: NDArray[np.int64]
    digital_batch: torch.Tensor
    gain_tensor: torch.Tensor
    offset_tensor: torch.Tensor
    arrow_column: pa.ListArray
    arrow_requests: list[tuple[int, int]]


def parse_arguments() -> ExperimentConfig:
    """Parse settings; reject invalid sizes, existing output or unavailable CUDA."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edf", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if min(arguments.batch_size, arguments.length, arguments.repeats) < 1 or arguments.warmup < 0:
        parser.error("Sizes/repeats must be positive and warmup nonnegative")
    if arguments.batch_size < 3:
        parser.error("Batch-size must be at least 3 to include boundary windows")
    if arguments.output.exists():
        parser.error("Output already exists; choose a new path to preserve results")
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; run with GPU access")
    return ExperimentConfig(
        edf_path=arguments.edf,
        device=device,
        batch_size=arguments.batch_size,
        window_length=arguments.length,
        repeats=arguments.repeats,
        warmup=arguments.warmup,
        output_path=arguments.output,
    )


def validate_continuous_edf(path: Path) -> None:
    """Reject unsupported formats, particularly EDF+D, before opening a reader.

    The fixed header identifies version and continuity. EDF+C annotation signals
    are handled by the readers rather than interpreted as voltage channels.
    """
    with path.open("rb") as stream:
        fixed_header = stream.read(256)
    if fixed_header[EDF_VERSION_SLICE].strip() != b"0" or fixed_header[EDF_RESERVED_SLICE].strip() not in (
        b"",
        b"EDF+C",
    ):
        raise ValueError("Only ordinary EDF or continuous EDF+C is supported")


def read_digital(path: Path, *, allow_out_of_range_samples: bool = False) -> DigitalRecording:
    """Read native samples and voltage calibration without resampling.

    Raises ValueError for discontinuities, mixed rates/lengths, unknown units,
    invalid calibration or samples outside declared digital bounds. Metadata
    extraction, validation and CPU layout conversion count as preparation.

    An explicit opt-in preserves samples outside declared bounds for source
    compatibility investigations. Calibration still uses the unchanged header;
    the caller must report those samples. Int16 representability remains required.
    """
    validate_continuous_edf(path)
    with pyedflib.EdfReader(str(path)) as reader:
        headers = reader.getSignalHeaders()
        sample_rates = reader.getSampleFrequencies()
        sample_counts = reader.getNSamples()
        if not headers or not np.all(sample_rates == sample_rates[0]) or not np.all(sample_counts == sample_counts[0]):
            raise ValueError("Mixed-rate or unequal-length signals require a separate experiment")
        try:
            unit_scales = np.array([VOLTAGE_UNIT_SCALES[header["dimension"]] for header in headers])
        except KeyError as error:
            raise ValueError("Only explicitly labelled voltage channels are supported") from error
        digital_min, digital_max, physical_min, physical_max = [
            np.array([header[key] for header in headers])
            for key in ("digital_min", "digital_max", "physical_min", "physical_max")
        ]
        if np.any(digital_min < -32768) or np.any(digital_max > 32767):
            raise ValueError("Expected int16 EDF samples")
        gain, offset = calibration_parameters(
            digital_min, digital_max, physical_min * unit_scales, physical_max * unit_scales
        )
        # EDF stores channel blocks; the benchmark consumes time-major arrays.
        samples = np.stack([reader.readSignal(channel, digital=True) for channel in range(len(headers))], axis=1)
        if np.any(samples < -32768) or np.any(samples > 32767):
            raise ValueError("Samples cannot be represented as int16")
        if not allow_out_of_range_samples and (np.any(samples < digital_min) or np.any(samples > digital_max)):
            raise ValueError("Samples exceed declared digital bounds")
        annotation_onsets, annotation_durations, annotation_descriptions = reader.readAnnotations()
        metadata = RecordingMetadata(
            headers=headers,
            sample_rate_hz=float(sample_rates[0]),
            sample_count=int(sample_counts[0]),
            start_datetime=reader.getStartdatetime().isoformat(),
            annotation_onset_s=annotation_onsets.tolist(),
            annotation_duration_s=annotation_durations.tolist(),
            annotation_description=annotation_descriptions.tolist(),
        )
    return DigitalRecording(samples.astype(np.int16), gain, offset, metadata)


def read_mne(path: Path) -> Float32Array:
    """Return (time, channels) volts using the existing builder's float32 path."""
    with mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR") as recording:
        return recording.get_data().T.astype(np.float32)


def validate_reference_metadata(reference: mne.io.BaseRaw, metadata: RecordingMetadata) -> None:
    """Require matching channel order/types, sample clock and annotation timing.

    Checks remain active under python -O. Mismatches abort the experiment rather
    than allowing timings for semantically different signals.
    """
    comparisons = {
        "channel order": reference.ch_names == [header["label"] for header in metadata.headers],
        "EEG channel types": all(kind == "eeg" for kind in reference.get_channel_types()),
        "sampling rate": reference.info["sfreq"] == metadata.sample_rate_hz,
        "sample count": reference.n_times == metadata.sample_count,
        "annotation descriptions": reference.annotations.description.tolist() == metadata.annotation_description,
    }
    start_datetime = reference.info["meas_date"]
    comparisons["recording start"] = (
        start_datetime is not None and start_datetime.replace(tzinfo=None).isoformat() == metadata.start_datetime
    )
    for field, matches in comparisons.items():
        if not matches:
            raise ValueError(f"MNE reference mismatch: {field}")
    np.testing.assert_allclose(
        reference.annotations.onset, metadata.annotation_onset_s, rtol=0, atol=ANNOTATION_TOLERANCE_SECONDS
    )
    np.testing.assert_allclose(
        reference.annotations.duration, metadata.annotation_duration_s, rtol=0, atol=ANNOTATION_TOLERANCE_SECONDS
    )


def build_arrow_column(physical_samples: Float32Array) -> pa.ListArray:
    """Represent one (time, channels) recording as nested Arrow numeric lists.

    Offsets reference numeric buffers, avoiding Python signal scalars. This
    setup is excluded from window-extraction measurements.
    """
    channel_count = physical_samples.shape[1]
    time_points = pa.ListArray.from_arrays(
        np.arange(0, physical_samples.size + 1, channel_count), pa.array(physical_samples.ravel())
    )
    return pa.ListArray.from_arrays([0, len(physical_samples)], time_points)


def prepare_experiment(config: ExperimentConfig) -> PreparedExperiment:
    """Read both representations and prepare shared deterministic batch inputs.

    Raises ValueError if windows cannot fit. Include first/last windows and a
    start near one second, while keeping all forced starts within bounds.
    """
    recording = read_digital(config.edf_path)
    with mne.io.read_raw_edf(str(config.edf_path), preload=True, verbose="ERROR") as reference:
        validate_reference_metadata(reference, recording.metadata)
        reference_volts = reference.get_data().T.copy()
    last_start = len(recording.samples) - config.window_length
    if last_start < 0 or config.batch_size < 3:
        raise ValueError("Window must fit recording; batch-size must be at least 3")
    window_starts = np.random.default_rng(0).integers(0, last_start + 1, size=config.batch_size)
    window_starts[:3] = [0, last_start, min(last_start, max(0, int(recording.metadata.sample_rate_hz) - 1))]
    window_indices = window_starts[:, None] + np.arange(config.window_length)
    cached_float32 = reference_volts.astype(np.float32)
    return PreparedExperiment(
        recording=recording,
        reference_volts=reference_volts,
        cached_float32=cached_float32,
        window_starts=window_starts,
        window_indices=window_indices,
        digital_batch=torch.from_numpy(recording.samples[window_indices]),
        gain_tensor=torch.from_numpy(recording.gain),
        offset_tensor=torch.from_numpy(recording.offset),
        arrow_column=build_arrow_column(cached_float32),
        arrow_requests=[(0, int(start)) for start in window_starts],
    )


def calibrate_numpy(samples: NDArray[np.int16], gain: Float32Array, offset: Float32Array) -> Float32Array:
    """Calibrate (..., channels) samples to volts with one float32 allocation.

    In-place multiply/add provide the NumPy baseline and preserve the input.
    Per-channel gain and offset broadcast over the final dimension.
    """
    calibrated = samples.astype(np.float32)
    np.multiply(calibrated, gain, out=calibrated)
    np.add(calibrated, offset, out=calibrated)
    return calibrated


def extract_arrow_windows(prepared: PreparedExperiment) -> torch.Tensor:
    """Copy selected Arrow windows into a writable float32 CPU tensor."""
    window_length, channel_count = prepared.digital_batch.shape[1:]
    return torch.from_numpy(
        extract_windows(prepared.arrow_column, prepared.arrow_requests, window_length, channel_count)
    )


def require_matching_signals(actual: NDArray, expected: NDArray) -> None:
    """Abort on sample/shape differences beyond the experiment's volt tolerance."""
    np.testing.assert_allclose(actual, expected, rtol=SIGNAL_RELATIVE_TOLERANCE, atol=SIGNAL_ABSOLUTE_TOLERANCE_VOLTS)


def validate_cpu_paths(prepared: PreparedExperiment) -> dict[str, float]:
    """Check full-record calibration and exact equality of cached Arrow windows."""
    recording = prepared.recording
    calibrated = calibrate(torch.from_numpy(recording.samples), prepared.gain_tensor, prepared.offset_tensor).numpy()
    require_matching_signals(calibrated, prepared.reference_volts)
    require_matching_signals(
        calibrate_numpy(recording.samples, recording.gain, recording.offset), prepared.reference_volts
    )
    np.testing.assert_array_equal(
        extract_arrow_windows(prepared).numpy(), prepared.cached_float32[prepared.window_indices]
    )
    return {"cpu_full_record_max_abs_error_volts": float(np.max(np.abs(calibrated - prepared.reference_volts)))}


def build_cpu_operations(config: ExperimentConfig, prepared: PreparedExperiment) -> BenchmarkOperations:
    """Bind CPU stages without moving preparation/gathering into batch stages."""

    def gather_cached_float32() -> torch.Tensor:
        return torch.from_numpy(prepared.cached_float32[prepared.window_indices])

    def calibrate_numpy_batch() -> Float32Array:
        return calibrate_numpy(prepared.digital_batch.numpy(), prepared.recording.gain, prepared.recording.offset)

    return {
        "preparation_mne_full_record_float32": partial(read_mne, config.edf_path),
        "preparation_pyedflib_full_record_int16": partial(read_digital, config.edf_path),
        "cached_numpy_float32_windows": gather_cached_float32,
        "cached_arrow_float32_windows": partial(extract_arrow_windows, prepared),
        "batch_numpy_cpu_calibration": calibrate_numpy_batch,
        "batch_torch_cpu_calibration": partial(
            calibrate, prepared.digital_batch, prepared.gain_tensor, prepared.offset_tensor
        ),
    }


def pin_and_transfer(samples: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Pin per call and enqueue a nonblocking transfer of a CPU tensor.

    Validation or the timing helper synchronizes the complete operation.
    """
    return samples.pin_memory().to(device, non_blocking=True)


def build_gpu_operations(device: torch.device, prepared: PreparedExperiment) -> BenchmarkOperations:
    """Bind CUDA stages with calibration parameters placed on the GPU once.

    Batch stages reuse gathered int16 samples; cached stages gather per call.
    Every operation includes pinning, transfer, allocation and any calibration.
    """
    gpu_gain = prepared.gain_tensor.to(device)
    gpu_offset = prepared.offset_tensor.to(device)

    def numpy_calibrate_pin_transfer() -> torch.Tensor:
        calibrated = calibrate_numpy(
            prepared.digital_batch.numpy(), prepared.recording.gain, prepared.recording.offset
        )
        return pin_and_transfer(torch.from_numpy(calibrated), device)

    def torch_calibrate_pin_transfer() -> torch.Tensor:
        calibrated = calibrate(prepared.digital_batch, prepared.gain_tensor, prepared.offset_tensor)
        return pin_and_transfer(calibrated, device)

    def int16_pin_transfer_gpu_calibrate() -> torch.Tensor:
        return calibrate(pin_and_transfer(prepared.digital_batch, device), gpu_gain, gpu_offset)

    def gather_float32_pin_transfer() -> torch.Tensor:
        windows = torch.from_numpy(prepared.cached_float32[prepared.window_indices])
        return pin_and_transfer(windows, device)

    def gather_int16_pin_transfer_gpu_calibrate() -> torch.Tensor:
        windows = torch.from_numpy(prepared.recording.samples[prepared.window_indices])
        return calibrate(pin_and_transfer(windows, device), gpu_gain, gpu_offset)

    return {
        "batch_numpy_calibrate_pin_transfer": numpy_calibrate_pin_transfer,
        "batch_torch_calibrate_pin_transfer": torch_calibrate_pin_transfer,
        "batch_int16_pin_transfer_gpu_calibrate": int16_pin_transfer_gpu_calibrate,
        "cached_numpy_float32_gather_pin_transfer": gather_float32_pin_transfer,
        "cached_int16_gather_pin_transfer_gpu_calibrate": gather_int16_pin_transfer_gpu_calibrate,
    }


def validate_gpu_paths(
    device: torch.device, prepared: PreparedExperiment, operations: BenchmarkOperations
) -> dict[str, float]:
    """Check all recording samples and each timed batch path against MNE volts."""
    calibrated = (
        calibrate(
            torch.from_numpy(prepared.recording.samples).to(device),
            prepared.gain_tensor.to(device),
            prepared.offset_tensor.to(device),
        )
        .cpu()
        .numpy()
    )
    require_matching_signals(calibrated, prepared.reference_volts)
    expected_windows = prepared.reference_volts[prepared.window_indices]
    for operation in operations.values():
        require_matching_signals(operation().cpu().numpy(), expected_windows)
    return {"cuda_full_record_max_abs_error_volts": float(np.max(np.abs(calibrated - prepared.reference_volts)))}


def measure_operations(
    operations: BenchmarkOperations, device: torch.device, config: ExperimentConfig
) -> StageMeasurements:
    """Measure named stages using the same warmup/repetition settings."""
    return {
        name: measure_stage(operation, device, config.warmup, config.repeats) for name, operation in operations.items()
    }


def run_benchmarks(
    config: ExperimentConfig, prepared: PreparedExperiment
) -> tuple[StageMeasurements, dict[str, float]]:
    """Validate before timing each device's stages; return latencies and errors."""
    checks = validate_cpu_paths(prepared)
    measurements = measure_operations(build_cpu_operations(config, prepared), torch.device("cpu"), config)
    if config.device.type == "cuda":
        gpu_operations = build_gpu_operations(config.device, prepared)
        checks.update(validate_gpu_paths(config.device, prepared, gpu_operations))
        measurements.update(measure_operations(gpu_operations, config.device, config))
    return measurements, checks


def build_report(
    config: ExperimentConfig,
    prepared: PreparedExperiment,
    measurements: StageMeasurements,
    checks: dict[str, float],
) -> dict[str, Any]:
    """Build provenance and results using the existing benchmark JSON keys.

    Environment inspection and hashing occur after timing. The Git revision
    identifies the checkout, not uncommitted experiment changes.
    """
    return {
        "scope": "One continuous equal-rate EEG recording; warm OS-cache preparation and in-memory batches. "
        "No cold disk, native window reader, resampling, training, or overlap benchmark.",
        "source": {
            "filename": config.edf_path.name,
            "sha256": hashlib.sha256(config.edf_path.read_bytes()).hexdigest(),
            "bytes": config.edf_path.stat().st_size,
        },
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "platform": platform.platform(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "torch": torch.__version__,
            "mne": mne.__version__,
            "pyedflib": pyedflib.__version__,
        },
        "device": str(config.device),
        "gpu": torch.cuda.get_device_name(config.device) if config.device.type == "cuda" else None,
        "torch_cpu_threads": torch.get_num_threads(),
        "shape": list(prepared.digital_batch.shape),
        "window_starts": prepared.window_starts.tolist(),
        "repeats": config.repeats,
        "warmup": config.warmup,
        "metadata": asdict(prepared.recording.metadata),
        "checks": {
            **checks,
            "channel_order_rates_units_annotations": "passed",
            "rtol": SIGNAL_RELATIVE_TOLERANCE,
            "atol_volts": SIGNAL_ABSOLUTE_TOLERANCE_VOLTS,
        },
        "stages": measurements,
    }


def write_report(output_path: Path, report: dict[str, Any]) -> None:
    """Save JSON exclusively and print the existing concise result summary.

    Exclusive creation also preserves an output created after CLI validation.
    """
    with output_path.open("x") as output:
        output.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output_path), "checks": report["checks"], "stages": report["stages"]}, indent=2))


def main() -> None:
    """Coordinate configuration, preparation, validated measurements and reporting."""
    config = parse_arguments()
    torch.set_num_threads(1)
    prepared = prepare_experiment(config)
    measurements, checks = run_benchmarks(config, prepared)
    report = build_report(config, prepared, measurements, checks)
    write_report(config.output_path, report)


if __name__ == "__main__":
    main()
