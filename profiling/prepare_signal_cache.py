"""Prepare both cache representations from an explicitly validated EDF source."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import mne
import numpy as np

from profiling.edf_signal_loading import read_digital, validate_reference_metadata
from profiling.signal_cache import SignalMetadata, checksum, write_signal_cache


def prepare_cache(
    original: Path,
    digital_source: Path,
    validation_report: Path,
    output: Path,
    *,
    accept_source_exceptions: bool = False,
) -> dict:
    """Validate provenance and all samples before storing a paired numeric cache.

    MNE reads the unchanged original. The integer reader uses the exact source
    identified by the previous correctness report. Header normalization or
    out-of-bounds samples require explicit acceptance; no repairs happen here.
    """
    started = time.perf_counter()
    report = json.loads(validation_report.read_text())
    if report["checks"]["source_metadata_and_selected_windows"] != "passed":
        raise ValueError("A successful source validation report is required")
    if (
        checksum(original) != report["source"]["sha256"]
        or checksum(digital_source) != report["digital_source"]["sha256"]
    ):
        raise ValueError("Source checksum does not match the validation report")
    has_exceptions = bool(report["normalization"]) or any(report["out_of_range_samples_per_channel"])
    if has_exceptions and not accept_source_exceptions:
        raise ValueError("Source exceptions require explicit --accept-source-exceptions")
    recording = read_digital(digital_source, allow_out_of_range_samples=report["out_of_range_samples_allowed"])
    with mne.io.read_raw_edf(str(original), preload=True, verbose="ERROR") as reference:
        validate_reference_metadata(reference, recording.metadata)
        physical = reference.get_data().T.astype(np.float32)
    metadata = SignalMetadata(
        channel_names=tuple(header["label"] for header in recording.metadata.headers),
        sample_rate_hz=recording.metadata.sample_rate_hz,
        provenance={
            "validation_report": report,
            "source_metadata": asdict(recording.metadata),
            "accepted_source_exceptions": accept_source_exceptions,
        },
    )
    write_signal_cache(output, physical, recording.samples, recording.gain, recording.offset, metadata)
    return {
        "output": str(output),
        "shape": list(physical.shape),
        "preparation_seconds": time.perf_counter() - started,
        "float32_file_bytes": (output / "float32.npy").stat().st_size,
        "int16_file_bytes": (output / "int16.npy").stat().st_size,
    }


def main() -> None:
    """Parse explicit preparation paths and report one-time preparation cost."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edf", type=Path)
    parser.add_argument("--digital-source", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--accept-source-exceptions", action="store_true")
    arguments = parser.parse_args()
    result = prepare_cache(
        arguments.edf,
        arguments.digital_source,
        arguments.validation_report,
        arguments.output_dir,
        accept_source_exceptions=arguments.accept_source_exceptions,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
