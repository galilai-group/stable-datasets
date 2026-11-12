import gzip
import struct

import numpy as np

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Image, Version
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, bulk_download


class OracleMNIST(BaseDatasetBuilder):
    """Oracle-MNIST Dataset using raw IDX files for ancient character classification."""

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://github.com/wm-bupt/oracle-mnist",
        citation="""@article{wang2024dataset,
                          title={A dataset of oracle characters for benchmarking machine learning algorithms},
                          author={Wang, Mei and Deng, Weihong},
                          journal={Scientific Data},
                          volume={11},
                          number={1},
                          pages={87},
                          year={2024}
                        }""",
        assets={
            "train_images": DownloadInfo(
                url="https://raw.githubusercontent.com/wm-bupt/oracle-mnist/"
                "b6cb263f84d39a8bbb94c5b806d846b600068924/data/oracle/train-images-idx3-ubyte.gz",
                checksum="sha256:46f51f7565fea2f78628032dfa410c938d9a6bcd3691cade43419d143228bec6",
            ),
            "train_labels": DownloadInfo(
                url="https://raw.githubusercontent.com/wm-bupt/oracle-mnist/"
                "b6cb263f84d39a8bbb94c5b806d846b600068924/data/oracle/train-labels-idx1-ubyte.gz",
                checksum="sha256:45c65b236f8e169e6bd0875270d84cb132753351ca1938ae7935523ebd1e95fc",
            ),
            "test_images": DownloadInfo(
                url="https://raw.githubusercontent.com/wm-bupt/oracle-mnist/"
                "b6cb263f84d39a8bbb94c5b806d846b600068924/data/oracle/t10k-images-idx3-ubyte.gz",
                checksum="sha256:ca5e631c21fdbf452f2af8cb410d8546456b4b9820881b8d77150767496bf099",
            ),
            "test_labels": DownloadInfo(
                url="https://raw.githubusercontent.com/wm-bupt/oracle-mnist/"
                "b6cb263f84d39a8bbb94c5b806d846b600068924/data/oracle/t10k-labels-idx1-ubyte.gz",
                checksum="sha256:e10758deebac47b5ce0a1fe87cea09d50cb797544ed7b7dbac61e5f709c074b0",
            ),
        },
    )

    def _info(self):
        return DatasetInfo(
            description="""The Oracle-MNIST dataset of noisy ancient Chinese oracle-bone characters from 10 categories,
            with a training set of 27,222 images and a test set of 3,000 images. Each image is 28x28 grayscale uint8.""",
            features=Features(
                {
                    "image": Image(),
                    "label": ClassLabel(
                        names=["big", "sun", "moon", "cattle", "next", "field", "not", "arrow", "time", "wood"]
                    ),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
            license="Apache-2.0",
        )

    def _candidate_splits(self):
        """Oracle-MNIST has four raw assets but two logical splits, so the cache is keyed by train/test."""
        return [Split.TRAIN, Split.TEST]

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
