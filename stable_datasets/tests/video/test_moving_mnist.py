import numpy as np

from stable_datasets.video.moving_mnist import MovingMNIST


def _write_synthetic_npy(path, num_sequences=3):
    # Source shape: (T=20, N, H=64, W=64) uint8
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(20, num_sequences, 64, 64), dtype=np.uint8)
    np.save(path, arr)
    return arr


def test_moving_mnist_test_split(tmp_path, monkeypatch):
    npy_path = tmp_path / "mnist_test_seq.npy"
    src = _write_synthetic_npy(npy_path, num_sequences=3)

    monkeypatch.setattr("stable_datasets.video.moving_mnist.download", lambda *a, **k: npy_path)

    ds = MovingMNIST(split="test", processed_cache_dir=tmp_path / "processed")
    assert len(ds) == 3

    sample = ds[0]
    assert set(sample.keys()) == {"video"}
    video = sample["video"]
    assert video.shape == (20, 64, 64, 1)
    assert video.dtype == np.uint8

    # Source layout is (T, N, H, W); index 0 should equal src[:, 0, :, :, None].
    np.testing.assert_array_equal(video, src[:, 0, :, :, None])
