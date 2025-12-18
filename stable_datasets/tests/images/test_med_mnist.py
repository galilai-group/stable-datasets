import numpy as np
from PIL import Image

from stable_datasets.images.med_mnist import MedMNIST


def test_med_mnist_variants_download_and_format():
    """
    Download/integration test for a small subset of MedMNIST variants.
    """

    variants = [
        "pathmnist",
        "chestmnist",
        "organmnist3d",
    ]

    for variant in variants:
        # Train split
        ds = MedMNIST(split="train", config_name=variant)
        assert len(ds) > 0, f"{variant} training dataset should not be empty."

        sample = ds[0]
        assert set(sample.keys()) == {"image", "label"}

        image = sample["image"]
        label = sample["label"]

        if variant.endswith("3d"):
            image_np = np.asarray(image)
            assert image_np.shape == (28, 28, 28), (
                f"{variant}: expected image shape (28, 28, 28), got {image_np.shape}."
            )
        else:
            assert isinstance(image, Image.Image), (
                f"{variant}: 'image' should be a PIL.Image object, got {type(image)}."
            )
            image_np = np.asarray(image)
            assert image_np.shape in [
                (28, 28),
                (28, 28, 3),
            ], f"{variant}: expected image shape (28, 28) or (28, 28, 3), got {image_np.shape}."

        if variant == "chestmnist":
            # multi-label
            label_np = np.asarray(label)
            assert label_np.ndim == 1, f"{variant}: expected 1D multi-label, got shape {label_np.shape}."
            assert set(np.unique(label_np).tolist()).issubset({0, 1}), (
                f"{variant}: labels must be 0/1, got {set(np.unique(label_np).tolist())}."
            )
        else:
            assert isinstance(label, int | np.integer), f"{variant}: label should be an int, got {type(label)}."

        # Validation + test splits
        ds_val = MedMNIST(split="validation", config_name=variant)
        assert len(ds_val) > 0, f"{variant} validation dataset should not be empty."

        ds_test = MedMNIST(split="test", config_name=variant)
        assert len(ds_test) > 0, f"{variant} test dataset should not be empty."
