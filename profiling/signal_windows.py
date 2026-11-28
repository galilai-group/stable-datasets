"""Sample-indexed request contract for the EEG storage experiment.

No storage, framework, GPU or label-policy dependencies belong in this module.
Time is converted to sample indices before issuing a request. Windows never
cross recordings; outputs preserve request and channel order.
"""

from dataclasses import asdict, dataclass
from math import isfinite
from operator import index

import numpy as np


@dataclass(frozen=True)
class Recording:
    """Source-order calibration to volts and dimensions of one recording."""

    n_samples: int
    rate_hz: float
    channels: tuple[str, ...]
    gain: tuple[float, ...]
    offset: tuple[float, ...]

    def __post_init__(self):
        object.__setattr__(self, "n_samples", index(self.n_samples))
        object.__setattr__(self, "rate_hz", float(self.rate_hz))
        object.__setattr__(self, "channels", tuple(self.channels))
        object.__setattr__(self, "gain", tuple(float(x) for x in self.gain))
        object.__setattr__(self, "offset", tuple(float(x) for x in self.offset))
        if index(self.n_samples) < 1 or not isfinite(self.rate_hz) or self.rate_hz <= 0:
            raise ValueError("Require positive sample count and finite positive sampling rate")
        if not self.channels or len(self.gain) != len(self.channels) or len(self.offset) != len(self.channels):
            raise ValueError("Channel order and calibration dimensions must agree")
        if any(not isinstance(name, str) or not name for name in self.channels):
            raise ValueError("Channel names must be nonempty strings")
        gain, offset = np.asarray(self.gain, dtype=np.float32), np.asarray(self.offset, dtype=np.float32)
        if not np.isfinite(gain).all() or not np.isfinite(offset).all() or np.any(gain == 0):
            raise ValueError("Calibration must be finite, nonzero and representable in float32")

    def to_dict(self):
        """Serialize metadata without source samples."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        """Restore immutable vectors from JSON."""
        return cls(value["n_samples"], value["rate_hz"], *(tuple(value[k]) for k in ("channels", "gain", "offset")))


@dataclass(frozen=True)
class Window:
    """A half-open sample interval in a single recording."""

    recording: int
    start: int
    length: int


def validate_windows(recordings: list[Recording], requests: list[Window]) -> None:
    """Reject invalid indices before any storage I/O."""
    for request in requests:
        row, start, length = index(request.recording), index(request.start), index(request.length)
        if row < 0 or row >= len(recordings) or start < 0 or length < 1:
            raise ValueError("Invalid recording or window interval")
        if start + length > recordings[row].n_samples:
            raise ValueError("Window extends beyond its recording")


def physical_windows(
    recordings: list[Recording], requests: list[Window], digital: list[np.ndarray]
) -> list[np.ndarray]:
    """Calibrate source-ordered digital windows to writable float32 volts."""
    from profiling.signal_calibration import calibrate_cpu

    validate_windows(recordings, requests)
    if len(digital) != len(requests):
        raise ValueError("Read cardinality differs from request cardinality")
    result = []
    for request, values in zip(requests, digital):
        meta = recordings[request.recording]
        if values.dtype != np.int16 or values.shape != (request.length, len(meta.channels)):
            raise ValueError("Digital read has an unexpected shape or dtype")
        result.append(
            calibrate_cpu(values, np.asarray(meta.gain, dtype=np.float32), np.asarray(meta.offset, dtype=np.float32))
        )
    return result
