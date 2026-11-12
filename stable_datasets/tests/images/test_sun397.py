import numpy as np
import pytest
from PIL import Image

from stable_datasets.images import SUN397


pytestmark = pytest.mark.large


def test_sun397_dataset():
    # Automatically download the full dataset
    sun397 = SUN397(split="train")

    # Load the first partition
    partition_01 = SUN397("partition_01")

    # Check that the dataset has the expected number of samples
    expected_num_train_samples = 108754
    assert len(sun397) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, got {len(sun397)}."
    )

    # Check the partition has the expected number of samples
    expected_num_partition_samples = 50 * 397
    assert len(partition_01["train"]) == expected_num_partition_samples, (
        f"Expected {expected_num_partition_samples} partition training samples, got {len(partition_01['train'])}."
    )
    assert len(partition_01["test"]) == expected_num_partition_samples, (
        f"Expected {expected_num_partition_samples} partition testing samples, got {len(partition_01['test'])}."
    )

    # create a set of test samples from the three splits
    test_samples = [
        sun397[0],
        sun397[len(sun397) // 2],
        sun397[len(sun397) - 1],
        partition_01["train"][0],
        partition_01["train"][25 * 397],
        partition_01["train"][50 * 397 - 1],
        partition_01["test"][0],
        partition_01["test"][25 * 397],
        partition_01["test"][50 * 397 - 1],
    ]

    for sample in test_samples:
        # Check that each sample has the keys "image" and "label"
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
        assert 0 <= label < 397, f"Label should be between 0 and 396, got {label}."

    print("All SUN397 dataset tests passed successfully!")
