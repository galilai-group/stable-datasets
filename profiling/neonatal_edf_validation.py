"""Validate selected neonatal EEG windows against the unmodified MNE source.

This correctness experiment does not benchmark or train. The source is read
unchanged by MNE. An explicitly requested derivative may clear an inconsistent
EDF+C marker for pyEDFlib; all other header bytes and signal bytes are preserved.
External seizure labels are three independent expert votes at 1 Hz, not EDF TALs.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pyedflib
import scipy
import torch
from numpy.typing import NDArray
from scipy.io import loadmat

from profiling.edf_signal_loading import (
    EDF_RESERVED_SLICE,
    read_digital,
    require_matching_signals,
    validate_reference_metadata,
    write_report,
)
from profiling.signal_loading import calibrate


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    """Hash a file in bounded memory for provenance and published MD5 checks."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, algorithm).hexdigest()


def normalize_continuous_header(source_path: Path, derivative_path: Path) -> dict[str, Any]:
    """Create an ordinary-EDF derivative of a mislabelled EDF+C source.

    This opt-in workaround accepts only EDF+C without an annotation channel.
    It clears the fixed header's reserved bytes 192:236 and copies every other
    byte verbatim. It is not a general repair tool and never overwrites files.
    Unchanged bytes do not establish continuity; signal/time checks against MNE
    remain necessary. The caller must report this limitation.
    """
    with source_path.open("rb") as source:
        fixed_header = source.read(256)
        if fixed_header[:8].strip() != b"0" or fixed_header[EDF_RESERVED_SLICE].strip() != b"EDF+C":
            raise ValueError("Normalization requires an EDF+C source")
        channel_count = int(fixed_header[252:256])
        header_size = int(fixed_header[184:192])
        if channel_count < 1 or header_size != 256 * (channel_count + 1):
            raise ValueError("Unexpected EDF header size")
        channel_header = source.read(header_size - 256)
        if len(channel_header) != header_size - 256:
            raise ValueError("Truncated EDF channel header")
        labels = [channel_header[index * 16 : (index + 1) * 16].strip() for index in range(channel_count)]
        if b"EDF Annotations" in labels:
            raise ValueError("Refusing to remove EDF+ semantics from an annotation-bearing source")
        normalized_header = fixed_header[:192] + b" " * 44 + fixed_header[236:]
        with derivative_path.open("xb") as derivative:
            derivative.write(normalized_header)
            derivative.write(channel_header)
            shutil.copyfileobj(source, derivative)
    # Verify the byte boundary explicitly, independently of reader behavior.
    with source_path.open("rb") as source, derivative_path.open("rb") as derivative:
        original_header, copied_header = source.read(256), derivative.read(256)
        if original_header[:192] != copied_header[:192] or original_header[236:] != copied_header[236:]:
            raise ValueError("Normalization changed unrelated fixed-header bytes")
        while original_block := source.read(1024 * 1024):
            if original_block != derivative.read(len(original_block)):
                raise ValueError("Normalization changed channel metadata or signal bytes")
        if derivative.read(1):
            raise ValueError("Normalization changed file size")
    return {"changed_byte_range": [192, 236], "all_other_bytes_identical": True}


def load_expert_labels(path: Path, subject_id: int) -> NDArray[np.int32]:
    """Return (3 experts, seconds) binary labels from the published MAT schema.

    Subject IDs are one-based. No voting, transposition or class collapse is
    applied. Malformed shapes and nonbinary labels fail before signal checks.
    """
    subjects = loadmat(path)["annotat_new"]
    if subjects.ndim != 2 or subjects.shape[0] != 1 or not 1 <= subject_id <= subjects.shape[1]:
        raise ValueError("Unexpected MAT subject layout or subject ID")
    labels = np.asarray(subjects[0, subject_id - 1])
    if labels.ndim != 2 or labels.shape[0] != 3 or labels.shape[1] == 0 or not np.isin(labels, [0, 1]).all():
        raise ValueError("Expected three experts with nonempty binary second-wise labels")
    return labels.astype(np.int32)


def select_window_starts(sample_count: int, sample_rate_hz: int, length: int, labels: NDArray[np.int32]) -> list[int]:
    """Select bounded edge, middle, second-boundary and seizure-transition windows.

    Each expert's first onset and offset is included when present. This samples
    annotation boundaries without treating three independent votes as consensus.
    """
    if sample_rate_hz < 1 or not 1 <= length <= sample_count:
        raise ValueError("Positive sample rate and a window fitting the recording are required")
    last_start = sample_count - length
    candidates = [0, last_start, sample_count // 2, sample_rate_hz - 1]
    for expert_labels in labels:
        transitions = np.diff(expert_labels)
        for direction in (1, -1):
            matches = np.flatnonzero(transitions == direction)
            if matches.size:
                candidates.append((int(matches[0]) + 1) * sample_rate_hz - 1)
    return sorted({min(last_start, max(0, start)) for start in candidates})


def validate_windows(
    original_path: Path,
    digital_path: Path,
    labels: NDArray[np.int32],
    device: torch.device,
    length: int,
    *,
    allow_out_of_range_samples: bool = False,
) -> dict[str, Any]:
    """Compare CPU/CUDA windows to original-source MNE float64 volts.

    pyEDFlib preparation currently reads the whole integer recording once. MNE
    uses preload=False and reads only the selected windows. No timing or native
    integer window-loading performance claim is made.
    """
    recording = read_digital(digital_path, allow_out_of_range_samples=allow_out_of_range_samples)
    digital_min = np.array([header["digital_min"] for header in recording.metadata.headers])
    digital_max = np.array([header["digital_max"] for header in recording.metadata.headers])
    outside_bounds = (recording.samples < digital_min) | (recording.samples > digital_max)
    sample_rate = int(recording.metadata.sample_rate_hz)
    if sample_rate != recording.metadata.sample_rate_hz:
        raise ValueError("This second-wise annotation check requires an integer sample rate")
    if recording.metadata.sample_count != labels.shape[1] * sample_rate:
        raise ValueError("Annotation coverage does not match recording duration exactly")
    starts = select_window_starts(recording.metadata.sample_count, sample_rate, length, labels)
    # Include a window containing each channel's first out-of-bounds sample, so
    # the opt-in behavior is checked numerically rather than merely counted.
    for channel in range(outside_bounds.shape[1]):
        invalid_samples = np.flatnonzero(outside_bounds[:, channel])
        if invalid_samples.size:
            starts.append(min(int(invalid_samples[0]), recording.metadata.sample_count - length))
    starts = sorted(set(starts))
    gain = torch.from_numpy(recording.gain).to(device)
    offset = torch.from_numpy(recording.offset).to(device)
    windows = []
    with mne.io.read_raw_edf(str(original_path), preload=False, verbose="ERROR") as reference:
        validate_reference_metadata(reference, recording.metadata)
        for start in starts:
            stop = start + length
            expected = reference.get_data(start=start, stop=stop).T
            samples = torch.from_numpy(recording.samples[start:stop])
            cpu_values = samples.numpy().astype(np.float32) * recording.gain + recording.offset
            actual_values = calibrate(samples.to(device), gain, offset).cpu().numpy()
            require_matching_signals(cpu_values, expected)
            require_matching_signals(actual_values, expected)
            # Half-open sample windows intersect seconds floor(start/fs)..floor((stop-1)/fs).
            first_second, stop_second = start // sample_rate, (stop - 1) // sample_rate + 1
            np.testing.assert_array_equal(
                reference.time_as_index([start / sample_rate, stop / sample_rate], use_rounding=True), [start, stop]
            )
            windows.append(
                {
                    "start_sample": start,
                    "stop_sample": stop,
                    "start_seconds": start / sample_rate,
                    "stop_seconds": stop / sample_rate,
                    "annotation_seconds_half_open": [first_second, stop_second],
                    "expert_votes": labels[:, first_second:stop_second].tolist(),
                    "cpu_max_abs_error_volts": float(np.max(np.abs(cpu_values - expected))),
                    "device_max_abs_error_volts": float(np.max(np.abs(actual_values - expected))),
                }
            )
    return {
        "metadata": asdict(recording.metadata),
        "expert_labels_shape": list(labels.shape),
        "expert_seizure_seconds": labels.sum(axis=1).tolist(),
        "out_of_range_samples_allowed": allow_out_of_range_samples,
        "out_of_range_samples_per_channel": outside_bounds.sum(axis=0).tolist(),
        "windows": windows,
    }


def parse_arguments() -> argparse.Namespace:
    """Require explicit source/subject/label paths and opt-in header normalization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edf", type=Path)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--subject-id", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--normalized-copy", type=Path)
    parser.add_argument("--allow-out-of-range-samples", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.edf.stem != f"eeg{arguments.subject_id}":
        parser.error("Source filename must match subject ID to avoid annotation mismatch")
    if arguments.output.exists() or (arguments.normalized_copy and arguments.normalized_copy.exists()):
        parser.error("Output/derivative already exists; choose new paths")
    if arguments.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; run with GPU access")
    return arguments


def run_validation(arguments: argparse.Namespace) -> dict[str, Any]:
    """Prepare source provenance, optional derivative and a correctness report."""
    labels = load_expert_labels(arguments.annotations, arguments.subject_id)
    normalization = None
    digital_path = arguments.edf
    if arguments.normalized_copy:
        normalization = normalize_continuous_header(arguments.edf, arguments.normalized_copy)
        digital_path = arguments.normalized_copy
    result = validate_windows(
        arguments.edf,
        digital_path,
        labels,
        torch.device(arguments.device),
        arguments.length,
        allow_out_of_range_samples=arguments.allow_out_of_range_samples,
    )
    return {
        "scope": "Selected neonatal EEG windows; correctness only, no training or throughput claim",
        "subject_id": arguments.subject_id,
        "source": {"filename": arguments.edf.name, "sha256": file_digest(arguments.edf)},
        "annotations": {"filename": arguments.annotations.name, "sha256": file_digest(arguments.annotations)},
        "digital_source": {"filename": digital_path.name, "sha256": file_digest(digital_path)},
        "normalization": normalization,
        "device": arguments.device,
        "gpu": torch.cuda.get_device_name(0) if arguments.device == "cuda" else None,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "mne": mne.__version__,
            "pyedflib": pyedflib.__version__,
            "scipy": scipy.__version__,
        },
        "checks": {"source_metadata_and_selected_windows": "passed"},
        "stages": {},
        **result,
    }


def main() -> None:
    """Coordinate one explicit neonatal correctness check and save its report."""
    arguments = parse_arguments()
    report = run_validation(arguments)
    write_report(arguments.output, report)


if __name__ == "__main__":
    main()
