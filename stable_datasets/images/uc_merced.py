import io
import zipfile

import datasets
from PIL import Image
from tqdm import tqdm

from stable_datasets.utils import BaseDatasetBuilder


class UCMerced(BaseDatasetBuilder):
    """UC Merced Land Use Dataset

    Abstract
    The UC Merced Land Use Dataset is a remote sensing image classification benchmark consisting of 2,100 aerial images extracted from the USGS National Map Urban Area Imagery collection. It covers 21 land use classes with 100 images per class. The images have a resolution of one foot per pixel.

    Context
    Land use classification is a fundamental task in remote sensing and earth observation. This dataset serves as a standard benchmark for evaluating computer vision models on aerial imagery, presenting challenges such as varying textures, scale, and object density.

    Content
    The dataset consists of:
    - **Images:** 2,100 RGB images (256x256 pixels).
    - **Classes:** 21 categories (e.g., agricultural, airplane, beach, buildings).
    - **Splits:** All 2,100 images are provided in the 'train' split (no official test split exists).
    """

    VERSION = datasets.Version("1.0.0")

    SOURCE = {
        "homepage": "http://weegee.vision.ucmerced.edu/datasets/landuse.html",
        "citation": """@inproceedings{yang2010bag,
                        title={Bag-of-visual-words and spatial extensions for land-use classification},
                        author={Yang, Yi and Newsam, Shawn},
                        booktitle={Proceedings of the 18th SIGSPATIAL International Conference on Advances in Geographic Information Systems},
                        pages={270--279},
                        year={2010}}""",
        "assets": {
            "train": "http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip",
        },
    }

    def _info(self):
        return datasets.DatasetInfo(
            description="UC Merced Land Use dataset with 21 aerial image categories.",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(names=self._labels()),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        """Generate examples from the ZIP archive."""
        with zipfile.ZipFile(data_path, "r") as z:
            for member in tqdm(z.infolist(), desc=f"Processing {split} set"):
                if member.is_dir():
                    continue

                filename = member.filename

                if filename.lower().endswith(".tif"):
                    try:
                        parts = filename.replace("\\", "/").split("/")
                        label_name = parts[-2]
                    except IndexError:
                        continue

                    if label_name in self._labels():
                        with z.open(member) as f:
                            image_bytes = f.read()

                        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                        yield (
                            filename,
                            {
                                "image": image,
                                "label": label_name,
                            },
                        )

    @staticmethod
    def _labels():
        return [
            "agricultural",
            "airplane",
            "baseballdiamond",
            "beach",
            "buildings",
            "chaparral",
            "denseresidential",
            "forest",
            "freeway",
            "golfcourse",
            "harbor",
            "intersection",
            "mediumresidential",
            "mobilehomepark",
            "overpass",
            "parkinglot",
            "river",
            "runway",
            "sparseresidential",
            "storagetanks",
            "tenniscourt",
        ]
