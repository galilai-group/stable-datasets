"""Independent MNE checks for EDF metadata, annotations and physical windows."""

from __future__ import annotations

import mne
import numpy as np
from numpy.typing import NDArray

from profiling.edf_source import NativeEdfReader
from profiling.signal_calibration import calibrate_cpu


SIGNAL_RELATIVE_TOLERANCE = 2e-6
SIGNAL_ABSOLUTE_TOLERANCE_VOLTS = 1e-10


def require_matching_signals(actual: NDArray, expected: NDArray) -> None:
    """Abort on sample/shape differences beyond the experiment's volt tolerance."""
    np.testing.assert_allclose(actual, expected, rtol=SIGNAL_RELATIVE_TOLERANCE, atol=SIGNAL_ABSOLUTE_TOLERANCE_VOLTS)


def read_mne_windows(reference: mne.io.BaseRaw, starts: np.ndarray, length: int) -> np.ndarray:
    """Read original-source windows as float64 volts without preloading."""
    return np.stack([reference.get_data(start=int(start), stop=int(start) + length).T for start in starts])


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
    physical = calibrate_cpu(digital, metadata.gain, metadata.offset)
    require_matching_signals(physical, expected)
    return {
        "selected_cpu_max_abs_error_volts": float(np.max(np.abs(physical - expected))),
        "selected_samples_outside_declared_bounds": int(
            np.count_nonzero((digital < metadata.digital_min) | (digital > metadata.digital_max))
        ),
        "channel_order_clock_start_annotations": "passed",
    }
