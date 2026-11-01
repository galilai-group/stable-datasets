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

    VERSION = Version("1.1.0")

    # J and Z are excluded because their signs require motion.
    LETTERS = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in "JZ"]

    ORIGINAL_LABELS = [i for i in range(26) if i not in (9, 25)]

    LABEL_MAP = {original: index for index, original in enumerate(ORIGINAL_LABELS)}

    KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/datamunge/sign-language-mnist"

    CSV_FILES = {
        "train": "sign_mnist_train.csv",
        "test": "sign_mnist_test.csv",
    }

    SOURCE = DatasetSource(
        homepage=("https://www.kaggle.com/datasets/datamunge/sign-language-mnist"),
        assets={
            "train": DownloadInfo(url=KAGGLE_URL),
            "test": DownloadInfo(url=KAGGLE_URL),
        },
        citation="""@misc{sign_language_mnist,
            title={Sign Language MNIST},
            author={DataMunge},
            howpublished={Kaggle},
            url={https://www.kaggle.com/datasets/datamunge/sign-language-mnist}
        }""",
        license="CC0-1.0",
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "Sign Language MNIST contains 28x28 grayscale "
                "images of 24 American Sign Language letters. "
                "J and Z are excluded because they require motion. "
                "The dataset is distributed through Kaggle under "
                "the CC0 Public Domain license."
            ),
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=self.LETTERS),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license=self.SOURCE["license"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        """Read the official Kaggle CSV from the dataset ZIP."""

        if split not in self.CSV_FILES:
            raise ValueError(f"Unsupported split: {split}")

        expected_file = self.CSV_FILES[split]

        with ZipFile(data_path, "r") as archive:
            names = archive.namelist()

            # Use only the root-level official CSV.
            # The Kaggle ZIP also contains duplicated CSV files
            # inside train/test subdirectories.
            if expected_file not in names:
                raise ValueError(f"Missing official CSV file: {expected_file}")

            with archive.open(expected_file) as file:
                text_file = io.TextIOWrapper(
                    file,
                    encoding="utf-8-sig",
                    newline="",
                )

                reader = csv.reader(text_file)

                expected_header = ["label"] + [f"pixel{i}" for i in range(1, 785)]

                header = next(reader, None)

                if header != expected_header:
                    raise ValueError(f"Unexpected CSV header in {expected_file}")

                for idx, row in enumerate(reader):
                    if len(row) != 785:
                        raise ValueError(f"Invalid row {idx} in {expected_file}")

                    try:
                        original_label = int(row[0])
                    except ValueError as exc:
                        raise ValueError(f"Invalid label in row {idx}") from exc

                    if original_label not in self.LABEL_MAP:
                        raise ValueError(f"Invalid label: {original_label}")

                    label = self.LABEL_MAP[original_label]

                    try:
                        values = np.asarray(
                            row[1:],
                            dtype=np.float64,
                        )
                    except ValueError as exc:
                        raise ValueError(f"Invalid pixel value in row {idx}") from exc

                    if not np.all(np.isfinite(values)):
                        raise ValueError(f"Invalid pixel value in row {idx}")

                    if np.any(values < 0) or np.any(values > 255):
                        raise ValueError(f"Pixel out of range in row {idx}")

                    if not np.all(values == np.floor(values)):
                        raise ValueError(f"Non-integer pixel value in row {idx}")

                    pixels = values.astype(np.uint8).reshape(28, 28)

                    image = PILImage.fromarray(pixels, mode="L")

                    yield (
                        idx,
                        {
                            "image": image,
                            "label": label,
                        },
                    )
