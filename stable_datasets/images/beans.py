import zipfile

from PIL import Image as PILImage

from stable_datasets.schema import ClassLabel, DatasetInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


class Beans(BaseDatasetBuilder):
    """Bean disease dataset for classification of three classes: Angular Leaf Spot, Bean Rust, and Healthy leaves."""

    VERSION = Version("1.0.0")

    # Single source-of-truth for dataset provenance + download locations.
    SOURCE = {
        "homepage": "https://github.com/AI-Lab-Makerere/ibean/",
        "assets": {
            "train": "https://storage.googleapis.com/ibeans/train.zip",
            "test": "https://storage.googleapis.com/ibeans/test.zip",
            "validation": "https://storage.googleapis.com/ibeans/validation.zip",
        },
        "citation": """@misc{makerere2020beans,
                         author = "{Makerere AI Lab}",
                         title = "{Bean Disease Dataset}",
                         year = "2020",
                         month = "January",
                         url = "https://github.com/AI-Lab-Makerere/ibean/"}""",
    }

    def _info(self):
        return DatasetInfo(
            description="""The IBeans dataset contains leaf images representing three classes:
                1) Healthy leaves, 2) Angular Leaf Spot, and 3) Bean Rust. Images are collected in Uganda for disease
                classification in the field.""",
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=["healthy", "angular_leaf_spot", "bean_rust"]),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license="MIT License",
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        with zipfile.ZipFile(data_path, "r") as archive:
            for file_name in archive.namelist():
                if file_name.endswith(".jpg"):
                    with archive.open(file_name) as file:
                        image_data = PILImage.open(file)
                        label_name = file_name.split("/")[1]
                        label = self.info.features["label"].str2int(label_name)
                        yield file_name, {"image": image_data, "label": label}
