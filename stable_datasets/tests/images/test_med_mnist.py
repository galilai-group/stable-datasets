import datasets
from PIL import Image
import numpy as np


def test_med_mnist_dataset():
    variants = [
        "pathmnist", "chestmnist", "retinamnist", "breastmnist",
        "organmnist3d", "nodulemnist3d"
    ]

    for variant in variants:
        print(f"Testing {variant}...")

        # Load the dataset for training split
        dataset = datasets.load_dataset("../../stable_datasets/images/med_mnist.py", name=variant, split="train", trust_remote_code=True)

        # Test 1: Check dataset length
        assert len(dataset) > 0, f"{variant} training dataset should not be empty."

        # Test 2: Validate structure of a sample
        sample = dataset[0]
        assert "image" in sample and "label" in sample, f"{variant} samples must have 'image' and 'label' keys."

        # Test 3: Validate image type and dimensions
        image = sample["image"]
        if variant.endswith("3d"):
            assert isinstance(image, list), f"{variant}: 'image' should be a list."
            image_np = np.array(image)
            assert image_np.shape == (28, 28, 28), f"{variant}: Image shape should be (28, 28, 28), got {image.shape}."
        else:
            assert isinstance(image, Image.Image), f"{variant}: 'image' should be a PIL.Image object."
            image_np = np.array(image)
            if image_np.ndim == 2:
                assert image_np.shape == (28, 28), f"{variant}: Image shape should be (28, 28), got {image_np.shape}."
            elif image_np.ndim == 3:
                assert image_np.shape == (28, 28, 3), f"{variant}: Image shape should be (28, 28, 3), got {image_np.shape}."

        # Test 4: Validate label type
        label = sample["label"]
        if variant == "chestmnist":
            assert isinstance(label, list), f"{variant}: 'label' should be a list."
            for l_ in label:
                assert l_ in [0, 1], f"{variant}: All label values should be 0 or 1."
        else:
            assert isinstance(label, int), f"{variant}: 'label' should be an integer, got {type(label)}."

        # Validate dataset splits (val/test)
        dataset_val = datasets.load_dataset("../../stable_datasets/images/med_mnist.py", name=variant, split="validation", trust_remote_code=True)
        assert len(dataset_val) > 0, f"{variant} val dataset should not be empty."
        dataset_test = datasets.load_dataset("../../stable_datasets/images/med_mnist.py", name=variant, split="test", trust_remote_code=True)
        assert len(dataset_test) > 0, f"{variant} test dataset should not be empty."

        print(f"{variant} tests passed successfully!\n")

    print("All variants passed the tests!")


if __name__ == "__main__":
    test_med_mnist_dataset()
