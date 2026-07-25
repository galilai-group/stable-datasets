import zipfile

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


class Beans(BaseDatasetBuilder):
    """Bean disease dataset for classification of three classes: Angular Leaf Spot, Bean Rust, and Healthy leaves."""

    VERSION = Version("1.0.0")

    # Single source-of-truth for dataset provenance + download locations.
    SOURCE = DatasetSource(
        homepage="https://github.com/AI-Lab-Makerere/ibean/",
        assets={
            "train": DownloadInfo(url="https://huggingface.co/datasets/AI-Lab-Makerere/beans/resolve/main/data/train.zip"),
            "test": DownloadInfo(url="https://huggingface.co/datasets/AI-Lab-Makerere/beans/resolve/main/data/test.zip"),
            "validation": DownloadInfo(url="https://huggingface.co/datasets/AI-Lab-Makerere/beans/resolve/main/data/validation.zip"),
        },
        citation="""@misc{makerere2020beans,
                         author = "{Makerere AI Lab}",
                         title = "{Bean Disease Dataset}",
                         year = "2020",
                         month = "January",
                         url = "https://github.com/AI-Lab-Makerere/ibean/"}""",
    )

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
                if not file_name.endswith(".jpg"):
                    continue
                image_bytes = archive.read(file_name)
                label_name = file_name.split("/")[1]
                label = self.info.features["label"].str2int(label_name)
                yield file_name, {"image": image_bytes, "label": label}
