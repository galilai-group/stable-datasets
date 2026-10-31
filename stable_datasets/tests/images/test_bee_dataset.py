import numpy as np
from PIL import Image

from stable_datasets.images import BeeDataset


def test_bee_dataset():
    # Load the training split.
    bee_dataset = BeeDataset(split="train")

    # Test 1: Check the expected number of samples.
    expected_num_samples = 7490
    assert len(bee_dataset) == expected_num_samples, (
        f"Expected {expected_num_samples} samples, got {len(bee_dataset)}."
    )

    # Test 2: Check that each sample has the expected keys.
    expected_keys = {
        "image",
        "varroa",
        "pollen",
        "wasps",
        "cooling",
    }
    sample = bee_dataset[0]
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}."

    # Test 3: Validate image type and format.
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."
    assert image.mode == "RGB", f"Image should be RGB, got mode {image.mode}."

    image_np = np.array(image)
    assert image_np.ndim == 3 and image_np.shape[2] == 3, (
        f"Image should have three RGB channels, got shape {image_np.shape}."
    )
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}."

    # Test 4: Validate image dimensions.
    expected_image_size = (150, 300)  # (width, height) in PIL format
    assert image.size == expected_image_size, f"Expected image size {expected_image_size}, got {image.size}."

    # Test 5: Validate all four labels are binary integers.
    label_names = ["varroa", "pollen", "wasps", "cooling"]
    for label_name in label_names:
        label = sample[label_name]
        assert isinstance(label, (int, np.integer)), f"Label '{label_name}' should be an integer, got {type(label)}."
        assert label in (0, 1), f"Label '{label_name}' should be 0 or 1, got {label}."

    # Test 6: Check label metadata.
    for label_name in label_names:
        assert bee_dataset.info.features[label_name].dtype == "int32", (
            f"Label '{label_name}' should have int32 metadata."
        )

    print("All BeeDataset tests passed successfully!")
