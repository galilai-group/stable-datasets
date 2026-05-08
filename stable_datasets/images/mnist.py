import gzip
import struct

import numpy as np

from stable_datasets.schema import ClassLabel, DatasetInfo, Features, Image, Version
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, bulk_download


class MNIST(BaseDatasetBuilder):
    """MNIST Dataset using raw IDX files for digit classification."""

    VERSION = Version("1.0.0")

    SOURCE = {
        "homepage": "http://yann.lecun.com/exdb/mnist/",
        "citation": """@misc{lecun1998mnist,
                          author={Yann LeCun and Corinna Cortes and Christopher J.C. Burges},
                          title={The MNIST database of handwritten digits},
                          year={1998},
                          url={http://yann.lecun.com/exdb/mnist/}
                        }""",
        "assets": {
            "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
            "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
            "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
            "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
        },
    }

    def _info(self):
        return DatasetInfo(
            description="""The MNIST database of handwritten digits, with a training set of 60,000 examples, and a test
            set of 10,000 examples.""",
            features=Features(
                {
                    "image": Image(),
                    "label": ClassLabel(num_classes=10),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _split_generators(self):
        source = self._source()
        assets = source["assets"]
        urls = list(assets.values())
        local_paths = bulk_download(urls, dest_folder=self._raw_download_dir)
        url_to_path = dict(zip(assets.keys(), local_paths))

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "images_path": url_to_path["train_images"],
                    "labels_path": url_to_path["train_labels"],
                },
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={
                    "images_path": url_to_path["test_images"],
                    "labels_path": url_to_path["test_labels"],
                },
            ),
        ]

    def _generate_examples(self, images_path, labels_path):
        with gzip.open(images_path, "rb") as img_file:
            _, num_images, rows, cols = struct.unpack(">IIII", img_file.read(16))
            images = np.frombuffer(img_file.read(), dtype=np.uint8).reshape(num_images, rows, cols)

        with gzip.open(labels_path, "rb") as lbl_file:
            _, num_labels = struct.unpack(">II", lbl_file.read(8))
            labels = np.frombuffer(lbl_file.read(), dtype=np.uint8)

        assert len(images) == len(labels), "Mismatch between image and label counts."

        for idx, (image, label) in enumerate(zip(images, labels)):
            yield idx, {"image": image, "label": int(label)}
