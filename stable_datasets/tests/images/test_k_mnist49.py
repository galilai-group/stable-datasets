import numpy as np
from PIL import Image

from stable_datasets.images import KMNIST49


NUM_CLASSES = 49


def test_kmnist49_dataset():
    # KMNIST49(split="train") automatically downloads and loads the dataset
    k49_train = KMNIST49(split="train")

    # Test 1: Check that the dataset has the expected number of samples
    expected_num_train_samples = 232365
    assert len(k49_train) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, got {len(k49_train)}."
    )

    # Test 2: Check that each sample has the keys "image" and "label"
    sample = k49_train[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}"

    # Test 3: Validate image type, shape and dtype
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."
    image_np = np.array(image)
    assert image_np.shape == (28, 28), f"Image should have shape (28, 28), got {image_np.shape}"
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}"

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    assert 0 <= label < NUM_CLASSES, f"Label should be between 0 and {NUM_CLASSES - 1}, got {label}."

    # Test 5: Validate class names; index 48 is the hiragana iteration mark (U+309D)
    names = k49_train.info.features["label"].names
    assert len(names) == NUM_CLASSES, f"Expected {NUM_CLASSES} class names, got {len(names)}."
    assert names[0] == "a" and names[47] == "n" and names[48] == "iteration_mark", (
        f"Unexpected class name order: {names[0]}, {names[47]}, {names[48]}"
    )

    # Test 6: Check the test split and that every class appears in it
    k49_test = KMNIST49(split="test")
    expected_num_test_samples = 38547
    assert len(k49_test) == expected_num_test_samples, (
        f"Expected {expected_num_test_samples} test samples, got {len(k49_test)}."
    )
    test_labels = set(k49_test.table.column("label").to_numpy().tolist())
    assert test_labels == set(range(NUM_CLASSES)), (
        f"Expected all {NUM_CLASSES} classes in the test split, got {len(test_labels)}."
    )
