import numpy as np
from PIL import Image

from stable_datasets.images import COIL100


def test_coil100_dataset():
    # Test 1: Check the dataset size and available split.
    ds = COIL100(split="train")
    assert len(ds) == 7200
    assert set(COIL100(split=None)) == {"train"}

    # Test 2: Check the sample fields.
    first, last = ds[0], ds[-1]
    assert set(first) == {"image", "label", "angle"}

    # Test 3: Check the image type, shape, and data type.
    assert isinstance(first["image"], Image.Image)
    assert np.asarray(first["image"]).shape == (128, 128, 3)
    assert np.asarray(first["image"]).dtype == np.uint8

    # Test 4: Check labels and viewing angles.
    assert isinstance(first["label"], int)
    assert isinstance(first["angle"], int)
    assert (first["label"], first["angle"]) == (0, 0)
    assert (last["label"], last["angle"]) == (99, 355)

    # Test 5: Check the object class names.
    assert ds.features["label"].names == [f"obj{object_id}" for object_id in range(1, 101)]

    # Test 6: Check all 72 views of each object appear in order.
    views = zip(ds.table.column("label").to_pylist(), ds.table.column("angle").to_pylist())
    assert list(views) == [(label, angle) for label in range(100) for angle in range(0, 360, 5)]
