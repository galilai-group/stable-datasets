import datasets
from PIL import Image
import numpy as np


def test_fgvc_aircraft_dataset():
    # Load the FGVC Aircraft dataset
    fgvc_aircraft = datasets.load_dataset("../../stable_datasets/images/fgvc_aircraft.py", split="train", trust_remote_code=True)

    # Test 1: Check the dataset size (ensure reasonable number of samples)
    assert len(fgvc_aircraft) == 3334, f"Expected 3400 samples, got {len(fgvc_aircraft)}."

    # Test 2: Check that each sample has the keys "image" and "variant"
    sample = fgvc_aircraft[0]
    assert "image" in sample and "variant" in sample, "Each sample must have 'image' and 'variant' keys."

    # Test 3: Validate image type (PIL.Image)
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."

    # Optionally: convert to numpy array to check shape if needed
    image_np = np.array(image)
    assert image_np.ndim == 3, f"Image should be a 3D array, got {image_np.ndim} dimensions."

    # Test 4: Validate variant type and value
    variant = sample["variant"]
    assert isinstance(variant, int), f"Variant should be a string, got {type(variant)}."
    assert 0 <= variant < 100, f"Variant should be in the range [0, 99], got {variant}."

    # Test 5: Check the existence of validation split
    fgvc_aircraft_val = datasets.load_dataset("../../stable_datasets/images/fgvc_aircraft.py", split="validation", trust_remote_code=True)
    assert len(fgvc_aircraft_val) == 3333, f"Expected 3333 samples, got {len(fgvc_aircraft_val)}."

    # Test 6: Check the existence of test split
    fgvc_aircraft_test = datasets.load_dataset("../../stable_datasets/images/fgvc_aircraft.py", split="test", trust_remote_code=True)
    assert len(fgvc_aircraft_test) == 3333, f"Expected 3333 samples, got {len(fgvc_aircraft_test)}."

    print("All tests passed successfully!")


if __name__ == "__main__":
    test_fgvc_aircraft_dataset()
