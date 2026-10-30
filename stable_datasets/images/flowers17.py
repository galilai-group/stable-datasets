import io
import os
import tarfile

import scipy.io
from PIL import Image as PILImage

from stable_datasets.schema import (
    BuilderConfig,
    ClassLabel,
    DatasetInfo,
    DatasetSource,
    DownloadInfo,
    Features,
    Version,
)
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, bulk_download


class Flowers17(BaseDatasetBuilder):
    """Oxford Flowers-17 classification dataset.

    This set contains images of flowers belonging to 17 different categories.
    The images were acquired by searching the web and taking pictures. There are
    80 images for each category.

    There are three official partitions, each contains 680 training, 340 validation, and 340 test images.
    Select a partition with config_name="partition1", "partition2", or "partition3".
    """

    VERSION = Version("1.0.0")
    BUILDER_CONFIGS = [
        BuilderConfig(name="partition1", description="Official partition 1"),
        BuilderConfig(name="partition2", description="Official partition 2"),
        BuilderConfig(name="partition3", description="Official partition 3"),
    ]
    DEFAULT_CONFIG_NAME = "partition1"

    SOURCE = DatasetSource(
        homepage="https://www.robots.ox.ac.uk/~vgg/data/flowers/17/",
        citation="""@InProceedings{Nilsback06,
            author = {Maria-Elena Nilsback and Andrew Zisserman},
            title = {A Visual Vocabulary for Flower Classification},
            booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
            volume = {2},
            pages = {1447--1454},
            year = {2006},
        }""",
        assets={
            "images": DownloadInfo(url="https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz"),
            "splits": DownloadInfo(url="https://www.robots.ox.ac.uk/~vgg/data/flowers/17/datasplits.mat"),
        },
    )

    def _info(self):
        return DatasetInfo(
            description="Oxford Flowers-17 dataset with 17 classes and three official partitions.",
            features=Features(
                {
                    "image": ImageFeature(encode_format="JPEG"),
                    "label": ClassLabel(names=self._labels()),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _candidate_splits(self):
        return [Split.TRAIN, Split.VALIDATION, Split.TEST]

    def _split_generators(self):
        source = self._source()
        asset_keys = ["images", "splits"]
        local_paths = bulk_download([source["assets"][key] for key in asset_keys], dest_folder=self._raw_download_dir)
        path_map = dict(zip(asset_keys, local_paths))

        return [
            SplitGenerator(name=Split.TRAIN, gen_kwargs={"path_map": path_map, "split": "train"}),
            SplitGenerator(name=Split.VALIDATION, gen_kwargs={"path_map": path_map, "split": "validation"}),
            SplitGenerator(name=Split.TEST, gen_kwargs={"path_map": path_map, "split": "test"}),
        ]

    def _generate_examples(self, path_map, split):
        partition = self.config.name[-1]
        split_key = {"train": "trn", "validation": "val", "test": "tst"}[split] + partition
        split_data = scipy.io.loadmat(path_map["splits"])
        ids_set = set(split_data[split_key].ravel())

        with tarfile.open(path_map["images"], "r:gz") as tar:
            for member in tar:
                if member.isfile() and member.name.endswith(".jpg"):
                    file_name = os.path.basename(member.name)
                    try:
                        image_id = int(file_name.split("_")[1].split(".")[0])
                    except (IndexError, ValueError):
                        continue

                    if image_id in ids_set:
                        f = tar.extractfile(member)
                        if f is None:
                            continue

                        image_bytes = f.read()
                        image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")

                        # IDs are one-based, with 80 consecutive images per class.
                        label = (image_id - 1) // 80
                        yield image_id, {"image": image, "label": label}

    @staticmethod
    def _labels():
        """Flower names in image-ID block order, corresponding to labels 0-16."""
        return [
            "daffodil",
            "snowdrop",
            "lily of the valley",
            "bluebell",
            "crocus",
            "iris",
            "tiger lily",
            "tulip",
            "fritillary",
            "sunflower",
            "daisy",
            "colt's foot",
            "dandelion",
            "cowslip",
            "buttercup",
            "windflower",
            "pansy",
        ]
