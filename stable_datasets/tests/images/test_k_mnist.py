import datasets
from PIL import Image


def test_k_mnist_dataset():
    # Load the KMNIST dataset
    kmnist = datasets.load_dataset("../../stable_datasets/images/k_mnist.py", name="kmnist", split="train", trust_remote_code=True)

    # Test 1: Check dataset size
    assert len(kmnist) == 60000, f"Expected 60000 training samples, got {len(kmnist)}."

    # Test 2: Check data structure
    sample = kmnist[0]
    assert "image" in sample and "label" in sample, "Each sample must have 'image' and 'label' keys."
    assert isinstance(sample["image"], Image.Image), "Image must be a PIL.Image object."
    assert isinstance(sample["label"], int), f"Label should be an integer, got {type(sample['label'])}."

    # Test 3: Validate image properties
    image = sample["image"]
    assert image.mode == "L", f"Image mode should be 'L' for grayscale, got {image.mode}."
    assert image.size == (28, 28), f"Image size should be (28, 28), got {image.size}."

    # Test 4: Check label range
    assert 0 <= sample["label"] < 10, f"Label should be in range [0, 9], got {sample['label']}."

    # Test 5: Validate test split size
    kmnist_test = datasets.load_dataset("../../stable_datasets/images/k_mnist.py", name="kmnist", split="test", trust_remote_code=True)
    assert len(kmnist_test) == 10000, f"Expected 10000 test samples, got {len(kmnist_test)}."

    print("All tests passed successfully!")


if __name__ == "__main__":
    test_k_mnist_dataset()
