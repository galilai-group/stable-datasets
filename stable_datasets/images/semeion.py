import numpy as np
from PIL import Image as PILImage

from stable_datasets.schema import (
    ClassLabel,
    DatasetInfo,
    DatasetSource,
    DownloadInfo,
    Features,
    Version,
)
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


class Semeion(BaseDatasetBuilder):
    """Semeion handwritten digit classification dataset.

    Contains 1,593 grayscale images of handwritten digits (0-9).
    Each image is 16x16 pixels, represented by 256 binary pixel
    values in the original dataset.

    The original dataset has no predefined train/test split.
    This implementation uses NumPy's random generator with seed 42
    to create a reproducible 80/20 split:
    - train: 1,274 images
    - test: 319 images

    Each example returns a PIL grayscale image and an integer
    label between 0 and 9.
    """

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://archive.ics.uci.edu/dataset/178/semeion+handwritten+digit",
        license="CC BY 4.0",
        citation="""@misc{semeion_handwritten_digit_178,
        title        = {{Semeion Handwritten Digit}},
        year         = {1998},
        howpublished = {UCI Machine Learning Repository},
        note         = {{DOI}: https://doi.org/10.24432/C5SC8V}
        }""",
        assets={
            "train": DownloadInfo(
                url="https://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data"
            ),
            "test": DownloadInfo(url="https://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data"),
        },
    )

    def _info(self):
        return DatasetInfo(
            description=("Semeion handwritten digits dataset containing 1,593 grayscale 16x16 images of digits 0-9."),
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=[str(i) for i in range(10)]),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
            license=self.SOURCE["license"],
        )

    def _generate_examples(self, data_path, split):
        """Generate images and labels from the Semeion text file."""

        data = np.loadtxt(data_path, dtype=np.float32)

        if data.shape != (1593, 266):
            raise ValueError(f"Unexpected Semeion shape: {data.shape}")

        pixels = data[:, :256]
        targets = data[:, 256:]

        if not np.all((pixels == 0) | (pixels == 1)):
            raise ValueError("Pixel values must be binary")

        if not np.all((targets == 0) | (targets == 1)):
            raise ValueError("Class indicators must be binary")

        if not np.all(targets.sum(axis=1) == 1):
            raise ValueError("Each example must have exactly one class")

        # Deterministic 80/20 split
        indices = np.random.default_rng(42).permutation(len(data))
        train_size = int(0.8 * len(data))

        if split == "train":
            selected_indices = indices[:train_size]
        elif split == "test":
            selected_indices = indices[train_size:]
        else:
            raise ValueError(f"Unknown split: {split}")

        for idx in selected_indices:
            pixel_row = pixels[idx]
            target_row = targets[idx]

            image_array = (pixel_row.reshape(16, 16) * 255).astype(np.uint8)

            image = PILImage.fromarray(image_array, mode="L")
            label = int(np.argmax(target_row))

            yield (
                int(idx),
                {
                    "image": image,
                    "label": label,
                },
            )
