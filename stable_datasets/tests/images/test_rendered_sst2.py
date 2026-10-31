import numpy as np
import pytest
from PIL import Image

from stable_datasets.images import RenderedSST2


@pytest.mark.parametrize(
    "split, expected_num_samples, expected_counts",
    [
        ("train", 6920, [3310, 3610]),
        ("validation", 872, [428, 444]),
        ("test", 1821, [912, 909]),
    ],
)
def test_rendered_sst2_dataset(split, expected_num_samples, expected_counts):
    # RenderedSST2 automatically downloads and loads the requested split
    rendered_sst2 = RenderedSST2(split=split)

    # Test 1: Check that the split has the expected number of samples
    assert len(rendered_sst2) == expected_num_samples, (
        f"Expected {expected_num_samples} {split} samples, got {len(rendered_sst2)}."
    )

    # Test 2: Check that each sample has the keys "image" and "label"
    sample = rendered_sst2[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}"

    # Test 3: Validate image type, size, shape, and dtype
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."
    assert image.size == (448, 448), f"Image should have size (448, 448), got {image.size}"
    assert image.mode == "RGB", f"Image should be RGB, got {image.mode}."
    image_np = np.array(image)
    assert image_np.shape == (448, 448, 3), f"Image should have shape (448, 448, 3), got {image_np.shape}"
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}"

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    assert label in {0, 1}, f"Label should be 0 or 1, got {label}."

    # Test 5: Check class names in label order
    expected_names = ["negative", "positive"]
    assert rendered_sst2.info.features["label"].names == expected_names

    # Test 6: Read the Arrow label column without decoding images
    labels = rendered_sst2.table.column("label").to_numpy()
    assert np.all((labels >= 0) & (labels < 2)), "All labels should be 0 or 1."
    counts = np.bincount(labels, minlength=2).tolist()
    assert counts == expected_counts, f"Expected {split} class counts {expected_counts}, got {counts}."

    print(f"All Rendered SST2 {split} dataset tests passed successfully!")
