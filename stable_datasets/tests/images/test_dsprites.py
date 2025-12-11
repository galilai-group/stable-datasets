import datasets
import numpy as np
from PIL import Image


def test_dsprites_dataset(split="train"):
    # Load the DSprites dataset
    dsprites = datasets.load_dataset(
        "../../stable_datasets/images/dsprites.py",
        split=split,
        trust_remote_code=True
    )

    # Define the expected number of samples for train/test split
    if split == "train":
        expected_num_samples = round(737280 * 0.7)  # 70% of total
    elif split == "test":
        expected_num_samples = round(737280 * 0.3)  # 30% of total
    else:
        raise ValueError(f"Unknown split: {split}")

    # Test 1: Check the expected number of samples
    assert len(dsprites) == expected_num_samples, f"Expected {expected_num_samples} samples in {split} split, got {len(dsprites)}."

    # Test 2: Validate sample keys
    sample = dsprites[0]
    expected_keys = {"image", "orientation", "shape", "scale", "color", "position_x", "position_y"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}."

    # Test 3: Validate image type and shape
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."
    image_np = np.array(image)  # Convert to NumPy array to validate shape and dtype
    assert image_np.shape == (64, 64), f"Expected image shape (64, 64), got {image_np.shape}."
    assert image_np.dtype == np.uint8, f"Expected image dtype uint8, got {image_np.dtype}."

    # Test 4: Validate "orientation" field
    orientation = sample["orientation"]
    assert isinstance(orientation, float), f"Orientation should be a float, got {type(orientation)}."
    assert 0.0 <= orientation <= 2 * np.pi + 1e-5, f"Orientation out of range: {orientation}."

    # Test 5: Validate "shape" field
    shape = sample["shape"]
    assert isinstance(shape, int), f"Shape should be an integer, got {type(shape)}."
    assert 0 <= shape <= 2, f"Shape out of range: {shape}. Expected values are 0 (square), 1 (ellipse), or 2 (heart)."

    # Test 6: Validate "scale" field
    scale = sample["scale"]
    assert isinstance(scale, float), f"Scale should be a float, got {type(scale)}."
    assert 0.5 <= scale <= 1.0, f"Scale out of range: {scale}."

    # Test 7: Validate "color" field
    color = sample["color"]
    assert isinstance(color, int), f"Color should be an integer, got {type(color)}."
    assert color == 0, f"Unexpected color value: {color}. Expected value is 0 (white)."

    # Test 8: Validate "position_x" and "position_y" fields
    position_x = sample["position_x"]
    position_y = sample["position_y"]
    assert isinstance(position_x, float), f"Position_x should be a float, got {type(position_x)}."
    assert isinstance(position_y, float), f"Position_y should be a float, got {type(position_y)}."
    assert 0.0 <= position_x <= 1.0, f"Position_x out of range: {position_x}."
    assert 0.0 <= position_y <= 1.0, f"Position_y out of range: {position_y}."

    # Batch validation for dataset consistency
    for idx, example in enumerate(dsprites.select(range(100))):  # Test on first 100 samples
        assert set(example.keys()) == expected_keys, f"Sample {idx} missing expected keys."
        assert isinstance(example["image"], Image.Image), f"Sample {idx} has invalid image type."
        assert 0.0 <= example["orientation"] <= 2 * np.pi + 1e-5, f"Sample {idx} has invalid orientation."
        assert 0 <= example["shape"] <= 2, f"Sample {idx} has invalid shape."
        assert 0.5 <= example["scale"] <= 1.0, f"Sample {idx} has invalid scale."
        assert example["color"] == 0, f"Sample {idx} has invalid color."
        assert 0.0 <= example["position_x"] <= 1.0, f"Sample {idx} has invalid position_x."
        assert 0.0 <= example["position_y"] <= 1.0, f"Sample {idx} has invalid position_y."

    print(f"All DSprites dataset tests passed successfully for {split} split!")


if __name__ == "__main__":
    # Test train split
    test_dsprites_dataset(split="train")
    # Test test split
    test_dsprites_dataset(split="test")
