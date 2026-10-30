from collections import Counter

import numpy as np
import pytest
import requests
from PIL import Image

from stable_datasets.images import OlivettiFaces


def download_source_or_skip(source_info):
    """Use the shared download cache and skip unreachable sources."""
    from stable_datasets.utils import download

    try:
        return download(source_info)
    except RuntimeError as error:
        current = error

        while current is not None:
            if isinstance(
                current,
                (
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                ),
            ):
                pytest.skip(f"Dataset source unreachable: {error}")

            current = current.__cause__ or current.__context__

        raise


def test_olivetti_faces_dataset():
    """Validate the Olivetti Faces dataset and its splits."""
    train_ds = OlivettiFaces(split="train")
    test_ds = OlivettiFaces(split="test")

    assert len(train_ds) == 320
    assert len(test_ds) == 80

    train_labels = []
    test_labels = []

    for dataset, labels in [
        (train_ds, train_labels),
        (test_ds, test_labels),
    ]:
        for sample in dataset:
            assert set(sample.keys()) == {"image", "label"}

            image = sample["image"]
            label = sample["label"]

            assert isinstance(image, Image.Image)
            assert image.size == (64, 64)
            assert image.mode == "L"

            pixels = np.asarray(image)
            assert pixels.shape == (64, 64)
            assert pixels.dtype == np.uint8

            assert isinstance(label, int)
            assert 0 <= label < 40

            labels.append(label)

    assert set(train_labels) == set(range(40))
    assert set(test_labels) == set(range(40))

    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)

    assert all(train_counts[label] == 8 for label in range(40))
    assert all(test_counts[label] == 2 for label in range(40))


def test_olivetti_faces_source_consistency():
    """Compare all 400 examples against the cached original MAT file."""
    from scipy.io import loadmat

    source_info = OlivettiFaces.SOURCE["assets"]["train"]
    source_path = download_source_or_skip(source_info)

    raw_faces = loadmat(source_path)["faces"]
    assert raw_faces.shape == (4096, 400)

    expected_images = raw_faces.T.copy().reshape(400, 64, 64).transpose(0, 2, 1)

    train_ds = OlivettiFaces(split="train")
    test_ds = OlivettiFaces(split="test")

    train_indices = set()
    test_indices = set()

    for identity in range(40):
        for position in range(10):
            source_index = identity * 10 + position

            if position < 8:
                sample = train_ds[identity * 8 + position]
                train_indices.add(source_index)
            else:
                sample = test_ds[identity * 2 + position - 8]
                test_indices.add(source_index)

            np.testing.assert_array_equal(
                np.asarray(sample["image"]),
                expected_images[source_index],
            )
            assert int(sample["label"]) == identity

    assert len(train_indices) == 320
    assert len(test_indices) == 80
    assert train_indices.isdisjoint(test_indices)
    assert train_indices | test_indices == set(range(400))
