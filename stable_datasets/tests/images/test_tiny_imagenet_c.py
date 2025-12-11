import datasets
from PIL import Image
import numpy as np


def test_tiny_imagenet_c_dataset():
    # Load the Tiny ImageNet-C dataset
    tiny_imagenet_c = datasets.load_dataset("../../stable_datasets/images/tiny_imagenet_c.py", split="test", trust_remote_code=True)

    # Test 1: Check that each sample has the keys "image", "label", "corruption_name", and "corruption_level"
    sample = tiny_imagenet_c[0]
    expected_keys = {"image", "label", "corruption_name", "corruption_level"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}"

    # Test 2: Validate image type and shape
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."

    # Convert to numpy array to check shape
    image_np = np.array(image)
    assert image_np.shape == (64, 64, 3), f"Image should have shape (64, 64, 3), got {image_np.shape}"
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}"

    # Test 3: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    assert 0 <= label < 200, f"Label should be between 0 and 199, got {label}."

    # Test 4: Validate corruption_name and corruption_level
    corruption_name = sample["corruption_name"]
    corruption_level = sample["corruption_level"]
    corruptions = [
        "zoom_blur", "snow", "shot_noise", "pixelate", "motion_blur", "jpeg_compression",
        "impulse_noise", "glass_blur", "gaussian_noise", "frost", "fog", "elastic_transform",
        "defocus_blur", "contrast", "brightness"
    ]
    assert corruption_name in corruptions, f"Unexpected corruption_name: {corruption_name}"
    assert 1 <= corruption_level <= 5, f"corruption_level should be between 1 and 5, got {corruption_level}"

    print("All Tiny ImageNet-C tests passed successfully!")


if __name__ == "__main__":
    test_tiny_imagenet_c_dataset()
