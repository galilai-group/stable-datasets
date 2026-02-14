# import numpy as np
# from PIL import Image

# from stable_datasets.images.food101 import Food101


# def test_food101_dataset():
#     # Load training split
#     food101_train = Food101(split="train")

#     # Test 1: Check number of training samples
#     expected_num_train_samples = 75750
#     assert len(food101_train) == expected_num_train_samples, (
#         f"Expected {expected_num_train_samples} training samples, got {len(food101_train)}."
#     )

#     # Test 2: Check sample keys
#     sample = food101_train[0]
#     expected_keys = {"image", "label"}
#     assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}."

#     # Test 3: Validate image type
#     image = sample["image"]
#     assert isinstance(image, Image.Image), f"Image should be a PIL.Image.Image, got {type(image)}."

#     # Convert to numpy for basic sanity checks
#     image_np = np.array(image)
#     assert image_np.ndim == 3, f"Food-101 images should be HxWxC, got shape {image_np.shape}."
#     assert image_np.shape[2] == 3, f"Food-101 images should have 3 channels, got {image_np.shape[2]}."
#     assert image_np.dtype == np.uint8, f"Image dtype should be uint8, got {image_np.dtype}."

#     # Test 4: Validate label type and range
#     label = sample["label"]
#     assert isinstance(label, int), f"Label should be int, got {type(label)}."
#     assert 0 <= label < 101, f"Label should be in range [0, 100], got {label}."

#     # Test 5: Load and validate test split
#     food101_test = Food101(split="test")
#     expected_num_test_samples = 25250
#     assert len(food101_test) == expected_num_test_samples, (
#         f"Expected {expected_num_test_samples} test samples, got {len(food101_test)}."
#     )

#     print("All Food101 dataset tests passed successfully!")
