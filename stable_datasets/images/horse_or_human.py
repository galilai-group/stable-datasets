import io
import zipfile

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


CLASS_NAMES = ["horses", "humans"]


class HorseOrHuman(BaseDatasetBuilder):
    """Horses vs. Humans binary image classification dataset.

    Contains 300x300 RGB images belonging to two classes:
    horses and humans.

    The dataset preserves its original predefined splits:
    - train: 1,027 images
    - test: 256 images from the original validation archive

    Each example contains a PIL RGB image and an integer
    class label (0 for horses, 1 for humans).
    """

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://www.tensorflow.org/datasets/catalog/horses_or_humans",
        citation="""@misc{horses_or_humans,
        author = {Laurence Moroney},
        title = {Horses or Humans Dataset},
        month = feb,
        year = {2019},
        url = {http://laurencemoroney.com/horses-or-humans-dataset}
        }""",
        license="CC BY 2.0",
        assets={
            "train": DownloadInfo(url="https://storage.googleapis.com/learning-datasets/horse-or-human.zip"),
            "test": DownloadInfo(url="https://storage.googleapis.com/learning-datasets/validation-horse-or-human.zip"),
        },
    )

    def _info(self):
        return DatasetInfo(
            description="RGB images of horses and humans for binary image classification.",
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=CLASS_NAMES),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
            license=self.SOURCE["license"],
        )

    def _generate_examples(self, data_path, split):
        if split not in ("train", "test"):
            raise ValueError(f"Unknown split: {split}")

        class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}

        with zipfile.ZipFile(data_path, "r") as archive:
            for path in sorted(archive.namelist()):
                if not path.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                label_name = path.split("/")[-2].lower()

                if label_name not in class_to_idx:
                    continue

                with archive.open(path) as file:
                    image = PILImage.open(io.BytesIO(file.read())).convert("RGB")

                yield (
                    path,
                    {
                        "image": image,
                        "label": class_to_idx[label_name],
                    },
                )
