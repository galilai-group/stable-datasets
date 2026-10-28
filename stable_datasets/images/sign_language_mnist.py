import csv
import io
from zipfile import ZipFile

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


class SignLanguageMNIST(BaseDatasetBuilder):
    """Sign Language MNIST image classification dataset."""

    VERSION = Version("1.0.0")

    # J and Z are excluded because their signs require motion.
    LETTERS = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in "JZ"]
    ORIGINAL_LABELS = [i for i in range(26) if i not in (9, 25)]
    LABEL_MAP = {original: index for index, original in enumerate(ORIGINAL_LABELS)}

    SOURCE = DatasetSource(
        homepage="https://www.kaggle.com/datasets/datamunge/sign-language-mnist",
        assets={
            "train": DownloadInfo(
                url=("https://raw.githubusercontent.com/Elsaraf1/sign-language/main/sign_mnist_train.zip")
            ),
            "test": DownloadInfo(
                url=("https://raw.githubusercontent.com/Elsaraf1/sign-language/main/sign_mnist_test.zip")
            ),
        },
        citation="""@misc{sign_language_mnist,
            title={Sign Language MNIST},
            author={DataMunge},
            howpublished={Kaggle},
            url={https://www.kaggle.com/datasets/datamunge/sign-language-mnist}
        }""",
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "Sign Language MNIST contains 28x28 grayscale "
                "images of 24 American Sign Language letters. "
                "J and Z are excluded because they require motion."
            ),
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=self.LETTERS),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        """Read grayscale images and labels from the CSV ZIP."""

        with ZipFile(data_path, "r") as archive:
            csv_files = [name for name in archive.namelist() if name.lower().endswith(".csv")]

            if len(csv_files) != 1:
                raise ValueError(f"Expected one CSV file, found {len(csv_files)}")

            with archive.open(csv_files[0]) as file:
                text_file = io.TextIOWrapper(file, encoding="utf-8")
                reader = csv.reader(text_file)
                header = next(reader)

                expected_header = ["label"] + [f"pixel{i}" for i in range(1, 785)]

                if header != expected_header:
                    raise ValueError("Unexpected CSV header")

                for idx, row in enumerate(reader):
                    if len(row) != 785:
                        raise ValueError(f"Invalid row {idx}")

                    original_label = int(row[0])

                    if original_label not in self.LABEL_MAP:
                        raise ValueError(f"Invalid label: {original_label}")

                    label = self.LABEL_MAP[original_label]

                    try:
                        values = np.array(row[1:], dtype=np.float64)
                    except ValueError as exc:
                        raise ValueError(f"Invalid pixel value in row {idx}") from exc

                    if not np.all(np.isfinite(values)):
                        raise ValueError(f"Invalid pixel value in row {idx}")

                    if np.any(values < 0) or np.any(values > 255):
                        raise ValueError(f"Pixel out of range in row {idx}")

                    if not np.all(values == np.floor(values)):
                        raise ValueError(f"Non-integer pixel value in row {idx}")

                    pixels = values.astype(np.uint8).reshape(28, 28)
                    image = PILImage.fromarray(pixels)

                    yield (
                        idx,
                        {
                            "image": image,
                            "label": label,
                        },
                    )
