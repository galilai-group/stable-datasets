import os
import zipfile
from pathlib import Path

import datasets

from stable_datasets.utils import BaseDatasetBuilder


class RSSCN7(BaseDatasetBuilder):
    """Remote Sensing Scene Classification dataset with 7 scene categories.

    The `RSSCN7 <https://github.com/palewithout/RSSCN7>`_ dataset was created for
    the paper "Deep Learning Based Feature Selection for Remote Sensing Scene Classification"
    by Zou, Qin and Ni, Lihao and Zhang, Tong and Wang, Qian (IEEE GRSL 2015).

    It consists of 2,800 images extracted from Google Earth with a resolution of 400×400 pixels,
    covering 7 typical scene categories in remote sensing image: grassland, farmland,
    industrial and commercial regions, river and lake, forest, residential region, and parking lot.
    Each class contains 400 images.
    """

    VERSION = datasets.Version("1.0.0")

    # Single source-of-truth for dataset provenance + download locations.
    SOURCE = {
        "homepage": "https://github.com/palewithout/RSSCN7",
        "assets": {
            "train": "https://github.com/palewithout/RSSCN7/archive/refs/heads/master.zip"
        },
        "citation": """@article{zou2015deep,
                         title={Deep Learning Based Feature Selection for Remote Sensing Scene Classification},
                         author={Zou, Qin and Ni, Lihao and Zhang, Tong and Wang, Qian},
                         journal={IEEE Geoscience and Remote Sensing Letters},
                         volume={12},
                         number={11},
                         pages={2321--2325},
                         year={2015},
                         publisher={IEEE}}""",
    }

    def _info(self):
        return datasets.DatasetInfo(
            description="""The RSSCN7 dataset is a remote sensing scene classification dataset containing 2,800 images (400×400 pixels) across 7 scene categories: grassland (aGrass), farmland (bField), industrial and commercial regions (cIndustry), river and lake (dRiverLake), forest (eForest), residential region (fResident), and parking lot (gParking). Images were extracted from Google Earth. See https://github.com/palewithout/RSSCN7 for more information.""",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(
                        names=[
                            "aGrass",
                            "bField",
                            "cIndustry",
                            "dRiverLake",
                            "eForest",
                            "fResident",
                            "gParking",
                        ]
                    ),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        """Generate examples from the GitHub repository ZIP archive."""
        # Extract the ZIP file
        extract_dir = Path(data_path).parent / "rsscn7_extracted"
        if not extract_dir.exists():
            with zipfile.ZipFile(data_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

        # The GitHub archive creates a folder named "RSSCN7-master"
        rsscn7_dir = extract_dir / "RSSCN7-master"

        # Define the class folders
        class_names = ["aGrass", "bField", "cIndustry", "dRiverLake", "eForest", "fResident", "gParking"]

        idx = 0
        for class_name in class_names:
            class_path = rsscn7_dir / class_name
            if not class_path.exists():
                continue

            # Get all image files and sort them to ensure consistent ordering
            image_files = sorted([f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".png"))])

            for image_file in image_files:
                image_path = class_path / image_file
                yield idx, {"image": str(image_path), "label": class_name}
                idx += 1
