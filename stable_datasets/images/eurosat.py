import io

import pyarrow.parquet as pq
from PIL import Image as PILImage
from tqdm import tqdm

from stable_datasets.schema import (
    ClassLabel,
    DatasetInfo,
    Features,
    Version,
)
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


class EuroSAT(BaseDatasetBuilder):
    """EuroSAT RGB Dataset

    EuroSAT is a land use and land cover classification benchmark built from Sentinel-2 satellite
    imagery. The RGB version consists of 27,000 labeled 64x64 JPEG patches across 10 classes.
    The original release does not provide an official split; this builder uses the widely adopted
    ``google-research/remote_sensing_representations`` split (16,200 train / 5,400 validation /
    5,400 test) as mirrored by the ``timm/eurosat-rgb`` HuggingFace dataset.
    """

    VERSION = Version("1.0.0")

    SOURCE = {
        "homepage": "https://github.com/phelber/EuroSAT",
        "assets": {
            "train": "https://huggingface.co/datasets/timm/eurosat-rgb/resolve/main/data/train-00000-of-00001.parquet",
            "val": "https://huggingface.co/datasets/timm/eurosat-rgb/resolve/main/data/validation-00000-of-00001.parquet",
            "test": "https://huggingface.co/datasets/timm/eurosat-rgb/resolve/main/data/test-00000-of-00001.parquet",
        },
        "citation": """@article{helber2019eurosat,
            title={Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification},
            author={Helber, Patrick and Bischke, Benjamin and Dengel, Andreas and Borth, Damian},
            journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
            year={2019},
            publisher={IEEE}
        }""",
    }

    def _info(self):
        return DatasetInfo(
            description=(
                "EuroSAT RGB: 27,000 Sentinel-2 64x64 RGB patches covering 10 land use and land cover "
                "classes, with the google-research split of 16,200 train / 5,400 validation / 5,400 test."
            ),
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=self._labels()),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        """Read the parquet file for this split and yield (idx, example) pairs.

        Each parquet row has an ``image`` struct ``{"bytes": ..., "path": ...}`` (HF Image encoding)
        and an integer ``label``.
        """
        table = pq.read_table(data_path, columns=["image", "label"])
        images = table.column("image").to_pylist()
        labels = table.column("label").to_pylist()

        for idx, (img_entry, label) in enumerate(
            tqdm(zip(images, labels), total=len(images), desc=f"Processing EuroSAT {split}")
        ):
            raw = img_entry["bytes"] if isinstance(img_entry, dict) else img_entry
            image = PILImage.open(io.BytesIO(raw)).convert("RGB")
            yield idx, {"image": image, "label": int(label)}

    @staticmethod
    def _labels():
        return [
            "AnnualCrop",
            "Forest",
            "HerbaceousVegetation",
            "Highway",
            "Industrial",
            "Pasture",
            "PermanentCrop",
            "Residential",
            "River",
            "SeaLake",
        ]
