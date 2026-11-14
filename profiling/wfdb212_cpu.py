"""Fused compiled CPU baselines; one implementation compiled serial and parallel.

Use validated contiguous (..., 3) uint8 input and two float32 calibration values.
No fast-math: missing values and normal float32 calibration semantics are retained.
"""

import numpy as np
from numba import njit, prange


def _decode(packed, scale, baseline):
    """Unpack, sign-extend and calibrate each frame directly to its final output."""
    frames = packed.reshape(-1, 3)
    output = np.empty((len(frames), 2), dtype=np.float32)
    for frame in prange(len(frames)):
        middle = np.int32(frames[frame, 1])
        first = np.int32(frames[frame, 0]) | ((middle & 15) << 8)
        second = np.int32(frames[frame, 2]) | ((middle >> 4) << 8)
        for channel in range(2):
            digital = ((first if channel == 0 else second) ^ 2048) - 2048
            output[frame, channel] = (
                np.nan if digital == -2048 else (np.float32(digital) - baseline[channel]) * scale[channel]
            )
    return output.reshape(packed.shape[:-1] + (2,))


# Keep separate in-memory compilations: both variants share Python bytecode,
# so a shared on-disk function cache could mix serial and parallel artifacts.
decode_cpu_serial = njit(_decode)
decode_cpu_parallel = njit(parallel=True)(_decode)
