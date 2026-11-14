"""Measure original EDF -> selected windows -> CPU or GPU physical values.

Existing readers perform source interpretation. edfio lazily reads original EDF
records; MNE is the independent reference and CPU baseline. No processed cache,
header rewrite, resampling or custom GPU decoder is used by this experiment.
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import edfio
import mne
import numpy as np
import torch

from profiling.signal_utils import (
    VOLTAGE_UNIT_SCALES,
    calibrate,
    calibration_parameters,
    checksum,
    measure_stage,
    require_matching_signals,
)


@dataclass(frozen=True)
class NativeEdfMetadata:
    """Uniform sample clock and source-order calibration vectors in volts."""

    channel_names: tuple[str, ...]
    sample_rate_hz: float
    sample_count: int
    record_duration_s: float
    gain: np.ndarray
    offset: np.ndarray
    digital_min: np.ndarray
    digital_max: np.ndarray


class NativeEdfReader:
    """Read only requested signal records through edfio's public slice API.

    Supports continuous equal-rate voltage channels. Native values outside
    declared extrema are preserved, checked against MNE and reported. The reader
    does not change source headers or silently clip samples. edfio's lazy
    implementation maps source bytes and copies requested record/channel slices;
    this is CPU source loading, not GPU storage access.
    """

    def __init__(self, path: Path):
        with path.open("rb") as stream:
            if stream.read(8).strip() != b"0":
                raise ValueError("Expected ordinary 16-bit EDF encoding")
        self.recording = edfio.read_edf(path, lazy_load_data=True)
        signals = self.recording.signals
        if self.recording.reserved not in ("", "EDF+C"):
            raise ValueError("Only continuous EDF/EDF+C is supported")
        if not signals:
            raise ValueError("No signal channels")
        rates = np.array([signal.sampling_frequency for signal in signals])
        if not np.all(rates == rates[0]) or rates[0] <= 0:
            raise ValueError("Mixed sample rates require a separate resampling experiment")
        try:
            units = np.array([VOLTAGE_UNIT_SCALES[signal.physical_dimension] for signal in signals])
        except KeyError as error:
            raise ValueError("Only explicitly labelled voltage signals are supported") from error
        digital_min = np.array([signal.digital_min for signal in signals])
        digital_max = np.array([signal.digital_max for signal in signals])
        gain, offset = calibration_parameters(
            digital_min,
            digital_max,
            np.array([signal.physical_min for signal in signals]) * units,
            np.array([signal.physical_max for signal in signals]) * units,
        )
        sample_count = round(self.recording.duration * rates[0])
        if sample_count < 1 or not np.isclose(sample_count / rates[0], self.recording.duration):
            raise ValueError("Invalid recording duration")
        self.metadata = NativeEdfMetadata(
            tuple(signal.label for signal in signals),
            float(rates[0]),
            sample_count,
            self.recording.data_record_duration,
            gain,
            offset,
            digital_min,
            digital_max,
        )

    def read_windows(self, starts: np.ndarray, length: int) -> np.ndarray:
        """Return native int16 (batch, time, channels), preserving request order.

        Each call accesses the source recording again, including record slicing,
        CPU layout conversion and output allocation. No full digital/data signal
        property is accessed, so edfio does not materialize an entire recording.
        """
        if starts.ndim != 1 or not np.issubdtype(starts.dtype, np.integer):
            raise ValueError("Window starts must be an integer vector")
        if length < 1 or np.any(starts < 0) or np.any(starts > self.metadata.sample_count - length):
            raise ValueError("Window extends outside recording")
        output = np.empty((len(starts), length, len(self.metadata.channel_names)), dtype=np.int16)
        for batch_index, start in enumerate(starts):
            for channel, signal in enumerate(self.recording.signals):
                samples = signal.get_digital_slice(
                    int(start) / self.metadata.sample_rate_hz,
                    (int(start) + length) / self.metadata.sample_rate_hz,
                )
                if samples.dtype != np.int16 or samples.shape != (length,):
                    raise ValueError("Unexpected native dtype or window length")
                output[batch_index, :, channel] = samples
        return output


def read_mne_windows(reference: mne.io.BaseRaw, starts: np.ndarray, length: int) -> np.ndarray:
    """Read original-source windows as float64 volts without preloading."""
    return np.stack([reference.get_data(start=int(start), stop=int(start) + length).T for start in starts])


def calibrate_cpu(digital: np.ndarray, metadata: NativeEdfMetadata) -> np.ndarray:
    """Apply voltage calibration using one float32 allocation and in-place arithmetic."""
    physical = digital.astype(np.float32)
    physical *= metadata.gain
    physical += metadata.offset
    return physical


def transfer(samples: np.ndarray) -> torch.Tensor:
    """Include per-batch CPU pinning and H2D transfer in a complete source path."""
    return torch.from_numpy(samples).pin_memory().to("cuda", non_blocking=True)


def choose_windows(metadata: NativeEdfMetadata, batch_size: int, length: int) -> np.ndarray:
    """Include first/last windows and an EDF-record boundary among seeded requests."""
    last_start = metadata.sample_count - length
    if batch_size < 3 or last_start < 0:
        raise ValueError("Require at least three windows fitting the recording")
    starts = np.random.default_rng(0).integers(0, last_start + 1, size=batch_size)
    starts[:3] = [
        0,
        last_start,
        min(last_start, max(0, round(metadata.record_duration_s * metadata.sample_rate_hz) - 1)),
    ]
    return starts


def validate_source(reader: NativeEdfReader, reference: mne.io.BaseRaw, starts: np.ndarray, length: int) -> dict:
    """Validate native samples, metadata and EDF annotations against the unchanged MNE source."""
    metadata = reader.metadata
    if tuple(reference.ch_names) != metadata.channel_names or reference.info["sfreq"] != metadata.sample_rate_hz:
        raise ValueError("MNE channel order or sampling rate mismatch")
    if reference.n_times != metadata.sample_count:
        raise ValueError("MNE sample count mismatch")
    if reference.info["meas_date"].replace(tzinfo=None) != reader.recording.startdatetime:
        raise ValueError("MNE recording start mismatch")
    annotations = reader.recording.annotations
    np.testing.assert_allclose([entry.onset for entry in annotations], reference.annotations.onset, rtol=0, atol=1e-7)
    np.testing.assert_allclose(
        [entry.duration if entry.duration is not None else 0 for entry in annotations],
        reference.annotations.duration,
        rtol=0,
        atol=1e-7,
    )
    if [entry.text for entry in annotations] != reference.annotations.description.tolist():
        raise ValueError("MNE annotation text mismatch")
    digital = reader.read_windows(starts, length)
    expected = read_mne_windows(reference, starts, length)
    physical = calibrate_cpu(digital, metadata)
    require_matching_signals(physical, expected)
    return {
        "selected_cpu_max_abs_error_volts": float(np.max(np.abs(physical - expected))),
        "selected_samples_outside_declared_bounds": int(
            np.count_nonzero((digital < metadata.digital_min) | (digital > metadata.digital_max))
        ),
        "channel_order_clock_start_annotations": "passed",
    }


def build_operations(
    path: Path,
    reader: NativeEdfReader,
    reference: mne.io.BaseRaw,
    starts: np.ndarray,
    length: int,
    device: torch.device,
) -> tuple[dict, dict]:
    """Define source-inclusive paths; reused and reopen-per-batch stages stay separate."""

    def open_mne():
        with mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR"):
            pass

    def native_cpu():
        return calibrate_cpu(reader.read_windows(starts, length), reader.metadata)

    def mne_cpu():
        return read_mne_windows(reference, starts, length).astype(np.float32)

    cpu_operations = {
        "edfio_open_header_and_source_mapping": lambda: NativeEdfReader(path),
        "mne_open_header_and_close": open_mne,
        "mne_source_windows_to_cpu_float32": mne_cpu,
        "edfio_source_windows_to_cpu_int16": lambda: reader.read_windows(starts, length),
        "edfio_source_windows_to_cpu_float32": native_cpu,
    }
    if device.type != "cuda":
        return cpu_operations, {}
    gain = torch.from_numpy(reader.metadata.gain).to(device)
    offset = torch.from_numpy(reader.metadata.offset).to(device)

    def native_gpu():
        return calibrate(transfer(reader.read_windows(starts, length)), gain, offset)

    def reopened_mne():
        with mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR") as source:
            return transfer(read_mne_windows(source, starts, length).astype(np.float32))

    def reopened_native_cpu():
        source = NativeEdfReader(path)
        return transfer(calibrate_cpu(source.read_windows(starts, length), source.metadata))

    def reopened_native_gpu():
        source = NativeEdfReader(path)
        return calibrate(
            transfer(source.read_windows(starts, length)),
            torch.from_numpy(source.metadata.gain).to(device),
            torch.from_numpy(source.metadata.offset).to(device),
        )

    return cpu_operations, {
        "mne_source_windows_to_gpu_float32": lambda: transfer(mne_cpu()),
        "edfio_source_cpu_calibration_to_gpu": lambda: transfer(native_cpu()),
        "edfio_source_gpu_calibration": native_gpu,
        "mne_reopen_source_to_gpu": reopened_mne,
        "edfio_reopen_cpu_calibration_to_gpu": reopened_native_cpu,
        "edfio_reopen_gpu_calibration": reopened_native_gpu,
    }


def run_benchmark(path: Path, device: torch.device, batch_size: int, length: int, repeats: int) -> dict:
    """Check all complete paths before measuring warm-OS-cache source latency."""
    source_hash = checksum(path)
    torch.set_num_threads(1)
    reader = NativeEdfReader(path)
    starts = choose_windows(reader.metadata, batch_size, length)
    with mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR") as reference:
        checks = validate_source(reader, reference, starts, length)
        cpu_operations, gpu_operations = build_operations(path, reader, reference, starts, length, device)
        expected = read_mne_windows(reference, starts, length)
        for operation in gpu_operations.values():
            actual = operation().cpu().numpy()
            require_matching_signals(actual, expected)
        cpu_times = {
            name: measure_stage(operation, torch.device("cpu"), 3, repeats)
            for name, operation in cpu_operations.items()
        }
        gpu_times = {name: measure_stage(operation, device, 3, repeats) for name, operation in gpu_operations.items()}
    if checksum(path) != source_hash:
        raise ValueError("Source changed during benchmark")
    return {
        "scope": "Unmodified EDF -> source windows -> GPU; warm OS cache, no processed cache, no training",
        "source": {"name": path.name, "sha256": source_hash},
        "source_unchanged": True,
        "shape": [batch_size, length, len(reader.metadata.channel_names)],
        "window_starts": starts.tolist(),
        "sample_rate_hz": reader.metadata.sample_rate_hz,
        "sample_count": reader.metadata.sample_count,
        "reserved_marker": reader.recording.reserved,
        "channel_names": list(reader.metadata.channel_names),
        "checks": checks,
        "device": str(device),
        "torch_cpu_threads": 1,
        "repeats": repeats,
        "warmup": 3,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "versions": {
            "python": platform.python_version(),
            "edfio": edfio.__version__,
            "mne": mne.__version__,
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "code_sha256": checksum(Path(__file__)),
        "shared_helpers_sha256": checksum(Path(__file__).with_name("signal_utils.py")),
        "stages": {**cpu_times, **gpu_times},
    }


def main() -> None:
    """Run one original-source comparison and preserve prior output files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edf", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.batch_size < 3 or min(arguments.length, arguments.repeats) < 1:
        parser.error("Require batch-size >= 3 and positive length/repeats")
    if arguments.output.exists():
        parser.error("Output exists; choose a new filename")
    if arguments.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; run with GPU access")
    report = run_benchmark(
        arguments.edf, torch.device(arguments.device), arguments.batch_size, arguments.length, arguments.repeats
    )
    with arguments.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({"checks": report["checks"], "stages": report["stages"]}, indent=2))


if __name__ == "__main__":
    main()
