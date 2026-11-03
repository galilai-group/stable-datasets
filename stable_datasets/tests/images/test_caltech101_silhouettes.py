import numpy as np
from PIL import Image

from stable_datasets.images import Caltech101Silhouettes


def test_caltech101_silhouettes_dataset():
    # Official split sizes from caltech101_silhouettes_28_split1.mat
    expected_sizes = {"train": 4100, "validation": 2264, "test": 2307}
    splits = {name: Caltech101Silhouettes(split=name) for name in expected_sizes}

    # Test 1: Check expected number of samples per split
    for name, ds in splits.items():
        assert len(ds) == expected_sizes[name], f"Expected {expected_sizes[name]} {name} samples, got {len(ds)}."

    # Test 2: Check that each sample has the keys "image" and "label"
    expected_keys = {"image", "label"}
    for name, ds in splits.items():
        sample = ds[0]
        assert set(sample.keys()) == expected_keys, f"{name}: expected keys {expected_keys}, got {set(sample.keys())}"

    # Test 3: Validate image type, mode, shape, dtype and binary pixel values
    ds_train = splits["train"]
    image = ds_train[0]["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."
    assert image.mode == "L", f"Image should be grayscale (mode 'L'), got mode {image.mode}."
    image_np = np.array(image)
    assert image_np.shape == (28, 28), f"Image should have shape (28, 28), got {image_np.shape}"
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}"

    images = np.stack([np.array(ds_train[i]["image"]) for i in range(64)])
    assert set(np.unique(images).tolist()) == {0, 255}, "Silhouettes should be binary {0, 255}."
    # Source renders a black object on a white background, so background (255) dominates.
    assert images.mean() > 127, "Expected white background (255) to dominate; images may be inverted."

    # Test 4: Validate label type and 0-based range; every class appears in train
    label = ds_train[0]["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    for name, ds in splits.items():
        labels = ds.table.column("label").to_numpy()
        assert labels.min() == 0 and labels.max() == 100, (
            f"{name}: labels should span 0-100, got {labels.min()}-{labels.max()}."
        )
    train_labels = ds_train.table.column("label").to_numpy()
    assert len(np.unique(train_labels)) == 101, "Expected all 101 classes in the train split."
    assert np.bincount(train_labels).max() == 100, "Train split caps each class at 100 instances."

    # Test 5: Validate class names (verbatim from the source file) and a known sample
    names = ds_train.info.features["label"].names
    assert len(names) == 101, f"Expected 101 class names, got {len(names)}."
    assert names[0] == "Airplanes Side 2" and names[-1] == "yin yang"
    # First training example in the source file is a lamp (1-based label 57).
    assert names[label] == "lamp", f"Expected first train sample to be 'lamp', got '{names[label]}'."
    # Orientation: an upright lamp has a narrow stem in row 15; a transposed (row-major) reshape does not.
    stem_row = image_np[15]
    assert (stem_row[11:17] == 0).all() and (stem_row[:11] == 255).all() and (stem_row[17:] == 255).all(), (
        "Images appear transposed; expected Fortran-order reshape."
    )

    print("All Caltech-101 Silhouettes dataset tests passed successfully!")
