import io
from zipfile import ZipFile

import datasets
import numpy as np
from PIL import Image
from tqdm import tqdm

from stable_datasets.utils import BaseDatasetBuilder


class ArabicDigits(BaseDatasetBuilder):
    """Arabic Handwritten Digits Dataset."""

    VERSION = datasets.Version("1.0.0")

    # Single source-of-truth for dataset provenance + download locations.
    SOURCE = {
        "homepage": "https://github.com/mloey/Arabic-Handwritten-Digits-Dataset",
        "assets": {
            # Both splits come from the same CSV zip file
            "train": "https://raw.githubusercontent.com/mloey/Arabic-Handwritten-Digits-Dataset/master/Arabic%20Handwritten%20Digits%20Dataset%20CSV.zip",
            "test": "https://raw.githubusercontent.com/mloey/Arabic-Handwritten-Digits-Dataset/master/Arabic%20Handwritten%20Digits%20Dataset%20CSV.zip",
        },
        "citation": """@inproceedings{el2016cnn,
                        title={CNN for handwritten arabic digits recognition based on LeNet-5},
                        author={El-Sawy, Ahmed and Hazem, EL-Bakry and Loey, Mohamed},
                        booktitle={International conference on advanced intelligent systems and informatics},
                        pages={566--575},
                        year={2016},
                        organization={Springer}
                        }""",
    }

    def _info(self):
        return datasets.DatasetInfo(
            description="""Arabic Handwritten Digits Dataset containing 70,000 images of Arabic digits (0-9)
                           written by 700 participants. Images are 28x28 grayscale pixels.""",
            features=datasets.Features(
                {"image": datasets.Image(), "label": datasets.ClassLabel(names=[str(i) for i in range(10)])}
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        """Generate examples from the CSV zip archive."""
        # File names inside the zip
        if split == "train":
            images_file = "csvTrainImages 60k x 784.csv"
            labels_file = "csvTrainLabel 60k x 1.csv"
        else:  # test
            images_file = "csvTestImages.csv"
            labels_file = "csvTestLabel 10k x 1.csv"

        with ZipFile(data_path, "r") as archive:
            # Load images CSV (each row is 784 flattened pixels)
            with archive.open(images_file) as f:
                content = f.read().decode("utf-8")
                images = np.loadtxt(io.StringIO(content), delimiter=",", dtype=np.uint8)
                # Reshape from (N, 784) to (N, 28, 28) using Fortran order (MATLAB origin)
                images = images.reshape(-1, 28, 28, order="F")

            # Load labels CSV
            with archive.open(labels_file) as f:
                content = f.read().decode("utf-8")
                labels = np.loadtxt(io.StringIO(content), dtype=np.int32)

        # Generate examples
        for idx, (image, label) in enumerate(
            tqdm(zip(images, labels), total=len(labels), desc=f"Processing {split} set")
        ):
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image, mode="L")  # "L" for grayscale
            yield idx, {"image": pil_image, "label": int(label)}
