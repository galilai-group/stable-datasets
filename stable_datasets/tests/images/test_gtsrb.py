import numpy as np
from PIL import Image

from stable_datasets.images import GTSRB


def test_gtsrb_dataset():
    # Test 1: Check the official split sizes.
    datasets = GTSRB(split=None)
    assert set(datasets) == {"train", "test"}
    train, test = datasets["train"], datasets["test"]
    assert len(train) == 39209
    assert len(test) == 12630

    # Test 2: Check sample fields, image format, and label range.
    train_first, test_first = train[0], test[0]
    for sample in (train_first, test_first):
        assert set(sample) == {"image", "label"}
        assert isinstance(sample["image"], Image.Image)
        image = np.asarray(sample["image"])
        assert image.ndim == 3
        assert image.shape[2] == 3
        assert image.dtype == np.uint8
        assert isinstance(sample["label"], int)
        assert 0 <= sample["label"] < 43

    # Test 3: Check that original image dimensions are preserved.
    assert train_first["image"].size == (29, 30)
    assert test_first["image"].size == (53, 54)

    # Test 4: Check training labels and the separate test ground truth.
    assert (train_first["label"], train[-1]["label"]) == (0, 42)
    assert (test_first["label"], test[-1]["label"]) == (16, 10)

    # Test 5: Check class names and coverage in both splits.
    assert train.features["label"].names == [str(class_id) for class_id in range(43)]
    for dataset in (train, test):
        assert set(dataset.table.column("label").to_pylist()) == set(range(43))
