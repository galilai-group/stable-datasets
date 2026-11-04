import zipfile
from collections import defaultdict

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


class TrashNet(BaseDatasetBuilder):
    """TrashNet waste classification with 2,527 RGB images in six classes.

    Classes contain 403 cardboard, 501 glass, 410 metal, 594 paper,
    482 plastic, and 137 trash images. The resized images are 512 x 384.
    Within each class, archive paths are sorted lexicographically and
    every fifth image (zero-based indices 0, 5, 10, ...) goes to ``test``.
    The remaining images form ``train``: 2,019 training and 508 test images.
    This deterministic partition is defined by this builder, not an
    official split. macOS metadata entries are excluded.

    Homepage: https://github.com/garythung/trashnet
    """

    VERSION = Version("1.0.0")

    DATA_URL = "https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip"

    SOURCE = DatasetSource(
        homepage="https://github.com/garythung/trashnet",
        assets={
            "train": DownloadInfo(
                url=DATA_URL,
                filename="trashnet-resized.zip",
            ),
            "test": DownloadInfo(
                url=DATA_URL,
                filename="trashnet-resized.zip",
            ),
        },
        citation="""@misc{thung2016trashnet,
    author = {Thung, Gary and Yang, Mindy},
    title = {{TrashNet}: Dataset of Images of Trash},
    year = {2016},
    howpublished = {GitHub repository},
    url = {https://github.com/garythung/trashnet}
}""",
        license="MIT",
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "TrashNet contains images of household waste "
                "in six categories. The original images are "
                "divided into deterministic train/test splits."
            ),
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(
                        names=[
                            "cardboard",
                            "glass",
                            "metal",
                            "paper",
                            "plastic",
                            "trash",
                        ]
                    ),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license=self.SOURCE["license"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        class_names = self.info.features["label"].names
        groups = defaultdict(list)

        with zipfile.ZipFile(data_path, "r") as archive:
            for filename in archive.namelist():
                if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                parts = filename.replace("\\", "/").split("/")
                if len(parts) < 2 or "__MACOSX" in parts or parts[-1].startswith("._"):
                    continue

                label = parts[-2].lower()
                if label in class_names:
                    groups[label].append(filename)

            if not any(groups.values()):
                raise ValueError("No labeled TrashNet images found.")

            for label in class_names:
                label_id = self.info.features["label"].str2int(label)

                for index, filename in enumerate(sorted(groups[label])):
                    is_test = index % 5 == 0

                    if is_test != (split == "test"):
                        continue

                    yield (
                        filename,
                        {
                            "image": archive.read(filename),
                            "label": label_id,
                        },
                    )
