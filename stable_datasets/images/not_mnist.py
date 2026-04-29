import gzip
import struct

import numpy as np

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Image, Version
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, _default_dest_folder, bulk_download


class NotMNIST(BaseDatasetBuilder):
    """NotMNIST Dataset that contains images of letters A-J."""

    VERSION = Version("1.0.0")

    # Single source-of-truth for dataset provenance + download locations.
    SOURCE = DatasetSource(
        homepage="https://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html",
        assets={
            "train_images": DownloadInfo(
                url="https://github.com/davidflanagan/notMNIST-to-MNIST/raw/refs/heads/master/train-images-idx3-ubyte.gz"
            ),
            "train_labels": DownloadInfo(
                url="https://github.com/davidflanagan/notMNIST-to-MNIST/raw/refs/heads/master/train-labels-idx1-ubyte.gz"
            ),
            "test_images": DownloadInfo(
                url="https://github.com/davidflanagan/notMNIST-to-MNIST/raw/refs/heads/master/t10k-images-idx3-ubyte.gz"
            ),
            "test_labels": DownloadInfo(
                url="https://github.com/davidflanagan/notMNIST-to-MNIST/raw/refs/heads/master/t10k-labels-idx1-ubyte.gz"
            ),
        },
        citation="""@misc{bulatov2011notmnist,
                          author={Yaroslav Bulatov},
                          title={notMNIST dataset},
                          year={2011},
                          url={http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html}
                        }""",
    )

    def _info(self):
        return DatasetInfo(
            description="""A dataset that was created by Yaroslav Bulatov by taking some publicly available fonts and
            extracting glyphs from them to make a dataset similar to MNIST. There are 10 classes, with letters A-J.""",
            features=Features(
                {
                    "image": Image(),
                    "label": ClassLabel(names=self._labels()),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _split_generators(self):
        source = self._source()
        assets = source["assets"]

        # Get all URLs and download them
        asset_keys = list(assets.keys())
        download_dir = getattr(self, "_raw_download_dir", None)
        if download_dir is None:
            download_dir = _default_dest_folder()

        downloaded_paths = bulk_download([assets[key] for key in asset_keys], dest_folder=download_dir)
        path_by_key = dict(zip(asset_keys, downloaded_paths))

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "images_path": path_by_key["train_images"],
                    "labels_path": path_by_key["train_labels"],
                    "split": "train",
                },
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={
                    "images_path": path_by_key["test_images"],
                    "labels_path": path_by_key["test_labels"],
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
