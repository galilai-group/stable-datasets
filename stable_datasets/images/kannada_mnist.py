import gzip
import struct

import numpy as np

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Image, Version
from stable_datasets.splits import SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, bulk_download


_BASE_URL = "https://raw.githubusercontent.com/vinayprabhu/Kannada_MNIST/master/data/output_tensors/MNIST_format/"


class KannadaMNIST(BaseDatasetBuilder):
    """Kannada handwritten digits, using the original 60,000/10,000 split."""

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="https://github.com/vinayprabhu/Kannada_MNIST",
        assets={
            "train_images": DownloadInfo(
                url=_BASE_URL + "X_kannada_MNIST_train-idx3-ubyte.gz",
                checksum="sha256:11785e1e0846a852266d5323a51c1b495c207ad2acbeaaaf108211163481b212",
            ),
            "train_labels": DownloadInfo(
                url=_BASE_URL + "y_kannada_MNIST_train-idx1-ubyte.gz",
                checksum="sha256:ef11e1d3e378fefbe78d4c9913dc9617c83fc5f9240b090121fcc335b5642201",
            ),
            "test_images": DownloadInfo(
                url=_BASE_URL + "X_kannada_MNIST_test-idx3-ubyte.gz",
                checksum="sha256:4628f3fa7b1d32137449dd356be4601da0c28c420b79f1b745a5134504ca35ca",
            ),
            "test_labels": DownloadInfo(
                url=_BASE_URL + "y_kannada_MNIST_test-idx1-ubyte.gz",
                checksum="sha256:645db5902d290681c9e5b452428e24636963a50a3036c2201240351b88786272",
            ),
        },
        license="CC BY 4.0",
        citation="""@article{prabhu2019kannada,
            title={Kannada-MNIST: A new handwritten digits dataset for the Kannada language},
            author={Prabhu, Vinay Uday},
            journal={arXiv preprint arXiv:1908.01242}, year={2019}}
        """,
    )

    def _info(self):
        return DatasetInfo(
            description="70,000 28x28 grayscale images of handwritten Kannada digits.",
            features=Features({"image": Image(), "label": ClassLabel(names=[str(i) for i in range(10)])}),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE.homepage,
            citation=self.SOURCE.citation,
            license=self.SOURCE.license,
        )

    def _candidate_splits(self):
        return ["train", "test"]

    def _split_generators(self):
        assets = self.SOURCE.assets
        paths = dict(zip(assets, bulk_download(assets.values(), dest_folder=self._raw_download_dir)))
        return [
            SplitGenerator(
                name=split,
                gen_kwargs={
                    "data_path": paths[f"{split}_images"],
                    "labels_path": paths[f"{split}_labels"],
                    "split": split,
                },
            )
            for split in self._candidate_splits()
        ]

    def _generate_examples(self, data_path, labels_path, split):
        if split not in self._candidate_splits():
            raise ValueError(f"Unknown Kannada-MNIST split: {split!r}")
        with gzip.open(data_path, "rb") as stream:
            image_data = stream.read()
        with gzip.open(labels_path, "rb") as stream:
            label_data = stream.read()
        if len(image_data) < 16 or len(label_data) < 8:
            raise ValueError("Truncated Kannada-MNIST IDX header")
        magic, count, rows, columns = struct.unpack(">IIII", image_data[:16])
        label_magic, label_count = struct.unpack(">II", label_data[:8])
        if magic != 2051 or label_magic != 2049 or (rows, columns) != (28, 28):
            raise ValueError("Invalid Kannada-MNIST IDX magic number or image dimensions")
        if count != label_count or count == 0:
            raise ValueError("Kannada-MNIST image and label counts must match and be nonzero")
        if len(image_data) != 16 + count * rows * columns or len(label_data) != 8 + label_count:
            raise ValueError("Kannada-MNIST IDX payload length does not match its header")
        images = np.frombuffer(image_data, dtype=np.uint8, offset=16).reshape(count, rows, columns)
        labels = np.frombuffer(label_data, dtype=np.uint8, offset=8)
        if np.any(labels > 9):
            raise ValueError("Kannada-MNIST labels must be digits 0-9")
        for index, (image, label) in enumerate(zip(images, labels)):
            yield index, {"image": image, "label": int(label)}
