"""Read unchanged continuous, equal-rate EDF voltage signals through edfio.

This module owns metadata and native integer source access. It has no benchmark,
MNE reference, CUDA runtime, CLI or report-writing dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import edfio
import numpy as np

from profiling.signal_calibration import VOLTAGE_UNIT_SCALES, calibration_parameters


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
    declared extrema are preserved; independent checks live in edf_reference.
    The reader does not change source headers or silently clip samples. edfio's lazy
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
