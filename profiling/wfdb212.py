"""Native two-channel WFDB-212 reading; no resampling or processed cache.

Each frame packs two signed 12-bit samples into three bytes. Header semantics
come from WFDB. Unsupported channel layouts/skew/segments fail explicitly.
The reserved digital value -2048 becomes NaN only when converting to volts.
"""

from pathlib import Path

import numpy as np
import wfdb


def unpack_cpu(packed: np.ndarray) -> np.ndarray:
    """Decode (..., 3) uint8 packets to (..., 2) int16, retaining -2048 sentinels."""
    if packed.dtype != np.uint8 or packed.ndim < 2 or packed.shape[-1] != 3:
        raise ValueError("Expected uint8 packets with a final dimension of three")
    output = np.empty((*packed.shape[:-1], 2), dtype=np.int16)
    middle = packed[..., 1].astype(np.int16)
    output[..., 0] = packed[..., 0].astype(np.int16) | ((middle & 15) << 8)
    output[..., 1] = packed[..., 2].astype(np.int16) | ((middle >> 4) << 8)
    # XOR then subtraction performs signed 12-bit extension in int16 arithmetic.
    output ^= 2048
    output -= 2048
    return output


class Wfdb212Source:
    """Memory-map one original two-channel, single-file 212 recording.

    All channels require one sample/frame, zero skew and a common byte offset.
    Metadata setup is reusable; window gathering and decoding remain per batch.
    """

    def __init__(self, record: Path):
        self.record = record
        self.header = header = wfdb.rdheader(str(record))
        if not isinstance(header, wfdb.Record) or header.n_sig != 2 or header.fmt != ["212", "212"]:
            raise ValueError("Require a single-segment, two-channel WFDB-212 record")
        if len(set(header.file_name)) != 1 or header.samps_per_frame != [1, 1] or any(header.skew):
            raise ValueError("Multiple files, multiple samples/frame and channel skew are unsupported")
        offsets = [value or 0 for value in header.byte_offset]
        if (
            offsets[0] != offsets[1]
            or offsets[0] < 0
            or header.sig_len < 1
            or not np.isfinite(header.fs)
            or header.fs <= 0
        ):
            raise ValueError("Invalid byte offsets, recording length or sample clock")
        unit_scales = {"V": 1.0, "mV": 1e-3, "uV": 1e-6}
        try:
            self.units_to_volts = np.array([unit_scales[unit] for unit in header.units])
        except KeyError as error:
            raise ValueError("Unsupported physical units") from error
        gains = np.asarray(header.adc_gain, dtype=np.float64)
        baseline = np.asarray(header.baseline, dtype=np.float64)
        if not np.isfinite(gains).all() or np.any(gains == 0) or not np.isfinite(baseline).all():
            raise ValueError("Require finite calibration and nonzero ADC gains")
        self.scale = (self.units_to_volts / gains).astype(np.float32)
        self.baseline = baseline.astype(np.float32)
        if not np.isfinite(self.scale).all() or np.any(self.scale == 0) or not np.isfinite(self.baseline).all():
            raise ValueError("Calibration cannot be represented in float32")
        self.data_path = record.parent / header.file_name[0]
        if self.data_path.stat().st_size != offsets[0] + header.sig_len * 3:
            raise ValueError("Source byte count disagrees with two-channel 212 header")
        self.packets = np.memmap(
            self.data_path, dtype=np.uint8, mode="r", offset=offsets[0], shape=(header.sig_len, 3)
        )

    def gather(self, starts: np.ndarray, length: int) -> np.ndarray:
        """Copy ordered packed windows, including duplicates, from the original source."""
        if starts.ndim != 1 or not len(starts) or not np.issubdtype(starts.dtype, np.integer):
            raise ValueError("Require a nonempty integer start vector")
        if length < 1 or np.any(starts < 0) or np.any(starts > self.header.sig_len - length):
            raise ValueError("Window extends outside source")
        output = np.empty((len(starts), length, 3), dtype=np.uint8)
        for destination, start in zip(output, starts):
            np.copyto(destination, self.packets[int(start) : int(start) + length])
        return output

    def physical(self, digital: np.ndarray) -> np.ndarray:
        """Calibrate (..., 2) native integers to float32 volts, mapping missing data to NaN."""
        output = digital.astype(np.float32)
        output -= self.baseline
        output *= self.scale
        output[digital == -2048] = np.nan
        return output

    def reference(self, starts: np.ndarray, length: int, *, physical: bool) -> np.ndarray:
        """Read the same original windows via WFDB; physical reference uses float64 volts."""
        rows = []
        for start in starts:
            record = wfdb.rdrecord(
                str(self.record),
                sampfrom=int(start),
                sampto=int(start) + length,
                physical=physical,
                return_res=64 if physical else 16,
            )
            rows.append(record.p_signal * self.units_to_volts if physical else record.d_signal)
        return np.stack(rows)
