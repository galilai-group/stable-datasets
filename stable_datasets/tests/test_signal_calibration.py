"""Independent numerical checks for CPU/CUDA signal calibration."""

import numpy as np
import pytest
import torch

from profiling.signal_calibration import calibrate, calibrate_cpu, calibration_parameters


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_calibration_matches_independent_physical_reference(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA GPU not available")
    dmin = np.array([-32768, -2048, 0])
    dmax = np.array([32767, 2047, 1000])
    pmin = np.array([-200, 50, 5])
    pmax = np.array([300, -50, 20])
    values = np.stack([dmin, dmax, [0, -50, 400]]).astype(np.int16)[None]
    gain, offset = calibration_parameters(dmin, dmax, pmin, pmax)
    actual = (
        calibrate(
            torch.from_numpy(values).to(device), torch.from_numpy(gain).to(device), torch.from_numpy(offset).to(device)
        )
        .cpu()
        .numpy()
    )
    expected = (values.astype(np.float64) - dmin) * (pmax - pmin) / (dmax - dmin) + pmin
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-4)


@pytest.mark.parametrize(
    "bounds",
    [
        ([], [], [], []),
        ([0], [0], [-1], [1]),
        ([0], [10], [1], [1]),
        ([0], [10, 20], [-1], [1]),
        ([0], [10], [np.nan], [1]),
    ],
)
def test_invalid_calibration_metadata_is_rejected(bounds):
    with pytest.raises(ValueError):
        calibration_parameters(*bounds)


def test_calibration_rejects_wrong_dtype_and_channel_count():
    with pytest.raises(ValueError, match="int16"):
        calibrate(torch.zeros(2, 3), torch.ones(3), torch.zeros(3))
    with pytest.raises(ValueError, match="channel count"):
        calibrate(torch.zeros(2, 3, dtype=torch.int16), torch.ones(2), torch.zeros(2))


def test_cpu_calibration_preserves_input_and_matches_tensor_path():
    values = np.array([[[-32768, 0], [42, 32767]]], dtype=np.int16)
    gain, offset = calibration_parameters([-32768, -32768], [32767, 32767], [-1, -2], [3, 4])
    before = values.copy()
    actual = calibrate_cpu(values, gain, offset)
    expected = calibrate(torch.from_numpy(values), torch.from_numpy(gain), torch.from_numpy(offset)).numpy()
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(values, before)
