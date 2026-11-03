import numpy as np
import pytest
from PIL import Image

from stable_datasets.images import OracleMNIST


@pytest.mark.parametrize(
    "split, expected_num_samples, expected_counts",
    [
        ("train", 27222, [2433, 2765, 2668, 2614, 2610, 2328, 2710, 2360, 3335, 3399]),
        ("test", 3000, [300] * 10),
    ],
)
def test_oracle_mnist_dataset(split, expected_num_samples, expected_counts):
    # OracleMNIST automatically downloads and loads the requested split
    oracle_mnist = OracleMNIST(split=split)

    # Test 1: Check that the split has the expected number of samples
    assert len(oracle_mnist) == expected_num_samples, (
        f"Expected {expected_num_samples} {split} samples, got {len(oracle_mnist)}."
    )

    # Test 2: Check that each sample has the keys "image" and "label"
    sample = oracle_mnist[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}"

    # Test 3: Validate image type, size, shape, and dtype
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."
    assert image.size == (28, 28), f"Image should have size (28, 28), got {image.size}"
    image_np = np.array(image)
    assert image_np.shape == (28, 28), f"Image should have shape (28, 28), got {image_np.shape}"
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}"

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    assert 0 <= label < 10, f"Label should be between 0 and 9, got {label}."

    # Test 5: Check class names in label order
    expected_names = ["big", "sun", "moon", "cattle", "next", "field", "not", "arrow", "time", "wood"]
    assert oracle_mnist.info.features["label"].names == expected_names

    # Test 6: Read the Arrow label column without decoding images
    labels = oracle_mnist.table.column("label").to_numpy()
    assert np.all((labels >= 0) & (labels < 10)), "All labels should be between 0 and 9."
    counts = np.bincount(labels, minlength=10).tolist()
    assert counts == expected_counts, f"Expected {split} class counts {expected_counts}, got {counts}."

    print(f"All Oracle-MNIST {split} dataset tests passed successfully!")


def test_oracle_mnist_reuses_processed_cache(tmp_path, monkeypatch):
    kwargs = {"download_dir": tmp_path / "raw", "processed_cache_dir": tmp_path / "processed"}
    first = OracleMNIST(split="test", **kwargs)
    assert len(first) == 3000
    builder = object.__new__(OracleMNIST)
    OracleMNIST.__init__(builder)
    assert builder._candidate_splits() == ["train", "test"]

    def no_download(*args, **kwargs):
        pytest.fail("Loading a fully cached dataset must not download again")

    # Delete the raw files and block downloads: the split must load from the processed cache.
    for path in (tmp_path / "raw").rglob("*"):
        if path.is_file():
            path.unlink()
    monkeypatch.setattr("stable_datasets.images.oracle_mnist.bulk_download", no_download)
    monkeypatch.setattr("stable_datasets.utils.bulk_download", no_download)
    reopened = OracleMNIST(split="test", **kwargs)
    assert len(reopened) == 3000
    assert set(reopened[0].keys()) == {"image", "label"}
