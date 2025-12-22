import gzip
import struct

import datasets
import numpy as np

from stable_datasets.utils import BaseDatasetBuilder, _default_dest_folder, bulk_download


class NotMNIST(BaseDatasetBuilder):
    """NotMNIST Dataset that contains images of letters A-J."""

    VERSION = datasets.Version("1.0.0")

    # Single source-of-truth for dataset provenance + download locations.
    SOURCE = {
        "homepage": "https://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html",
        "assets": {
            "train_images": "https://github.com/davidflanagan/notMNIST-to-MNIST/raw/refs/heads/master/train-images-idx3-ubyte.gz",
            "train_labels": "https://github.com/davidflanagan/notMNIST-to-MNIST/raw/refs/heads/master/train-labels-idx1-ubyte.gz",
            "test_images": "https://github.com/davidflanagan/notMNIST-to-MNIST/raw/refs/heads/master/t10k-images-idx3-ubyte.gz",
            "test_labels": "https://github.com/davidflanagan/notMNIST-to-MNIST/raw/refs/heads/master/t10k-labels-idx1-ubyte.gz",
        },
        "citation": """@misc{bulatov2011notmnist,
                          author={Yaroslav Bulatov},
                          title={notMNIST dataset},
                          year={2011},
                          url={http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html}
                        }""",
    }

    def _info(self):
        return datasets.DatasetInfo(
            description="""A dataset that was created by Yaroslav Bulatov by taking some publicly available fonts and
            extracting glyphs from them to make a dataset similar to MNIST. There are 10 classes, with letters A-J.""",
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
        source = self._source()
        assets = source["assets"]

        # Get all URLs and download them
        urls = list(assets.values())
        download_dir = getattr(self, "_raw_download_dir", None)
        if download_dir is None:
            download_dir = _default_dest_folder()

        downloaded_paths = bulk_download(urls, dest_folder=download_dir)
        url_to_path = dict(zip(urls, downloaded_paths))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "images_path": url_to_path[assets["train_images"]],
                    "labels_path": url_to_path[assets["train_labels"]],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "images_path": url_to_path[assets["test_images"]],
                    "labels_path": url_to_path[assets["test_labels"]],
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, images_path, labels_path, split):
        # Read and parse the gzipped IDX files
        with gzip.open(images_path, "rb") as img_file:
            _, num_images, rows, cols = struct.unpack(">IIII", img_file.read(16))
            images = np.frombuffer(img_file.read(), dtype=np.uint8).reshape(num_images, rows, cols)

        with gzip.open(labels_path, "rb") as lbl_file:
            _, num_labels = struct.unpack(">II", lbl_file.read(8))
            labels = np.frombuffer(lbl_file.read(), dtype=np.uint8)

        assert len(images) == len(labels), "Mismatch between image and label counts."

        for idx, (image, label) in enumerate(zip(images, labels)):
            yield idx, {"image": image, "label": int(label)}

    @staticmethod
    def _labels():
        """Returns the list of NotMNIST labels (letters A-J)."""
        return ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
