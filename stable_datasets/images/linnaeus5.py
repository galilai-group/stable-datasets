import io

import datasets
from PIL import Image


# We use rarfile because the original source only provides a RAR archive.
# Ensure you have 'rarfile' installed and the 'unrar' system executable available.
try:
    import rarfile
except ImportError:
    raise ImportError("To use the Linnaeus 5 dataset, you must install the 'rarfile' package: pip install rarfile")

from stable_datasets.utils import BaseDatasetBuilder


class Linnaeus5(BaseDatasetBuilder):
    """Linnaeus 5 Dataset

    Abstract
    The Linnaeus 5 dataset contains 1,600 RGB images sized 256x256 pixels, categorized into 5 classes: berry, bird, dog, flower, and other (negative set). It was created to benchmark fine-grained classification and object recognition tasks.

    Context
    While many datasets focus on broad object categories (like CIFAR-10), Linnaeus 5 offers a focused challenge on specific natural objects plus a "negative" class ('other'). It serves as a good middle-ground benchmark between simple digit recognition (MNIST) and large-scale natural image classification (ImageNet).

    Content
    The dataset consists of:
    - **Images:** 8,000 color images (256x256 pixels).
    - **Classes:** 5 categories (berry, bird, dog, flower, other).
    - **Splits:** Pre-split into Training (1,200 images per class) and Test (400 images per class).
    """

    VERSION = datasets.Version("1.0.0")

    SOURCE = {
        "homepage": "http://chaladze.com/l5/",
        "citation": """@article{chaladze2017linnaeus,
                      title={Linnaeus 5 dataset for machine learning},
                      author={Chaladze, G and Kalatozishvili, L},
                      journal={chaladze.com},
                      year={2017}}""",
        "assets": {
            # Direct link to the 256x256 version (approx 433 MB)
            "data": "http://chaladze.com/l5/img/Linnaeus%205%20256X256.rar",
        },
    }

    def _info(self):
        return datasets.DatasetInfo(
            description="Linnaeus 5 dataset with 5 classes (berry, bird, dog, flower, other).",
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

    def _split_generators(self, dl_manager):
        """
        Download the RAR archive.
        Note: We do not use 'download_and_extract' because standard Python zip/tar
        tools generally do not support RAR. We pass the archive path to _generate_examples
        and handle extraction there using 'rarfile'.
        """
        source = self._source()
        url = source["assets"]["data"]

        archive_path = dl_manager.download(url)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "archive_path": archive_path,
                    "split_name": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "archive_path": archive_path,
                    "split_name": "test",
                },
            ),
        ]

    def _generate_examples(self, archive_path, split_name):
        """Iterate over the RAR archive and yield images matching the split."""

        # Expected structure inside RAR:
        # Linnaeus 5 256X256/train/berry/image.jpg
        # Linnaeus 5 256X256/test/berry/image.jpg

        with rarfile.RarFile(archive_path) as rf:
            # Filter the list to avoid iterating unnecessary files
            for member in rf.infolist():
                if member.isdir():
                    continue

                filename = member.filename
                # Simple check: path must contain the split name (e.g. "/train/") and be an image
                if f"/{split_name}/" in filename.lower() and filename.lower().endswith((".jpg", ".jpeg")):
                    # Path is likely: "Linnaeus 5 256X256/train/berry/123.jpg"
                    # We extract the label from the parent folder name
                    try:
                        # Split path components
                        parts = filename.replace("\\", "/").split("/")
                        # Label is the folder name immediately preceding the filename
                        label_name = parts[-2]
                    except IndexError:
                        continue

                    if label_name in self._labels():
                        # Read file bytes
                        with rf.open(member) as f:
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
        return ["berry", "bird", "dog", "flower", "other"]
