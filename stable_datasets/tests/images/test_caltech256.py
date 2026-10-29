import numpy as np
from PIL import Image

from stable_datasets.images import Caltech256


def test_caltech256_dataset():
    # Caltech256(split="train") automatically downloads and loads the dataset
    caltech256 = Caltech256(split="train")

    # Check that the dataset has the expected number of samples
    expected_num_train_samples = 30607
    assert len(caltech256) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, got {len(caltech256)}."
    )

    for idx in [0, len(caltech256) // 2, len(caltech256) - 1]:
        # Check that each sample has the keys "image" and "label"
        sample = caltech256[idx]
        expected_keys = {"image", "label"}
        assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}"

        # Validate image type
        image = sample["image"]
        assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."
        assert image.mode == "RGB", f"Image should have mode RGB, but got {image.mode}"

        # Convert to numpy for basic sanity checks
        image_np = np.asarray(image)
        assert image_np.dtype == np.uint8, f"Image dtype should be uint8, got {image_np.dtype}."
        assert image_np.ndim == 3, f"Image should have 3 dimensions (H, W, C), got shape {image_np.shape}"
        assert image_np.shape[2] == 3, f"Image should have 3 channels (RGB), got {image_np.shape[2]} channels"

        # Validate label type and range
        label = sample["label"]
        assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
        assert 0 <= label < 257, f"Label should be between 0 and 256, got {label}."

    print("All Caltech256 dataset tests passed successfully!")
