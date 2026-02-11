import os
import zipfile
from pathlib import Path

import datasets
from PIL import Image

from stable_datasets.utils import BaseDatasetBuilder


class RockPaperScissor(BaseDatasetBuilder):
    """Rock Paper Scissors dataset."""

    VERSION = datasets.Version("1.0.0")

    # Single source-of-truth for dataset provenance + download locations.
    SOURCE = {
        "homepage": "https://laurencemoroney.com/datasets.html",
        "assets": {
            "train": "https://storage.googleapis.com/download.tensorflow.org/data/rps.zip",
            "test": "https://storage.googleapis.com/download.tensorflow.org/data/rps-test-set.zip",
        },
        "citation": """@misc{laurence2019rock,
                         title={Rock Paper Scissors Dataset},
                         author={Laurence Moroney},
                         year={2019},
                         url={https://laurencemoroney.com/datasets.html}}""",
        "license": "CC By 2.0",
    }

    def _info(self):
        return datasets.DatasetInfo(
            description="""Rock Paper Scissors contains images from various hands, from different races, ages, and
                           genders, posed into Rock / Paper or Scissors and labeled as such.""",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(names=["rock", "paper", "scissors"]),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        """Generate examples from the extracted zip archive."""
        # Extract the zip file to a directory
        extract_dir = Path(data_path).parent / f"rock_paper_scissor_{split}"
        if not extract_dir.exists():
            with zipfile.ZipFile(data_path, "r") as zip_file:
                zip_file.extractall(extract_dir)

        # Walk through the extracted directory
        for root, _, files in os.walk(extract_dir):
            for file_name in files:
                if file_name.endswith(".png"):
                    label = os.path.basename(root)  # Folder name as label
                    file_path = os.path.join(root, file_name)
                    # Open image and ensure it is RGB
                    with open(file_path, "rb") as img_file:
                        image = Image.open(img_file).convert("RGB")
                        yield file_path, {"image": image, "label": label}
