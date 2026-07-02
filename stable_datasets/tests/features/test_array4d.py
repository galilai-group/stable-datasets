import numpy as np
import pytest

from stable_datasets.features import Array4D


def test_array4d_roundtrip():
    arr = np.arange(24, dtype=np.uint8).reshape(2, 3, 4, 1)
    feat = Array4D(shape=(2, 3, 4, 1), dtype="uint8")
    encoded = feat.encode(arr)
    assert isinstance(encoded, bytes)
    decoded = feat.format(encoded, format_type="numpy")
    assert decoded.shape == (2, 3, 4, 1)
    assert decoded.dtype == np.uint8
    np.testing.assert_array_equal(decoded, arr)


def test_array4d_none_passthrough():
    feat = Array4D(shape=(1, 1, 1, 1), dtype="uint8")
    assert feat.encode(None) is None
    assert feat.format(None, format_type="numpy") is None


def test_array4d_rejects_wrong_ndim():
    with pytest.raises(ValueError):
        Array4D(shape=(2, 3, 4), dtype="uint8")


def test_array4d_torch_format():
    torch = pytest.importorskip("torch")
    arr = np.arange(8, dtype=np.uint8).reshape(1, 2, 2, 2)
    feat = Array4D(shape=(1, 2, 2, 2), dtype="uint8")
    encoded = feat.encode(arr)
    out = feat.format(encoded, format_type="torch")
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 2, 2, 2)
