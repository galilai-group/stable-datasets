"""QMNIST reconstruction, with the original order and extended NIST labels."""

import gzip
import struct

import numpy as np

from stable_datasets.schema import (
    ClassLabel,
    DatasetInfo,
    DatasetSource,
    DownloadInfo,
    Features,
    Image,
    Value,
    Version,
)
from stable_datasets.splits import SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, download


_BASE_URL = "https://raw.githubusercontent.com/facebookresearch/qmnist/94061367ec2e099ea18c553c75727ecf849ddb9b/"
_METADATA_FIELDS = (
    "nist_hsf_series",
    "nist_writer_id",
    "nist_digit_index",
    "nist_class_code",
    "nist_global_index",
    "duplicate",
    "unused",
)


class QMNIST(BaseDatasetBuilder):
    """QMNIST train, full test, and the official test10k/test50k subsets.

    ``test10k`` and ``test50k`` partition ``test``; they are not three
    independent test sets. Images are unmodified 28 x 28 grayscale pixels.
    """

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="https://github.com/facebookresearch/qmnist",
        license="Data license not separately specified; repository software: BSD-3-Clause",
        citation="""@incollection{qmnist-2019,
  title={Cold Case: The Lost MNIST Digits},
  author={Chhavi Yadav and L{\\'e}on Bottou},
  booktitle={Advances in Neural Information Processing Systems 32},
  year={2019}, publisher={Curran Associates, Inc.}
}""",
        assets={
            "train_images": DownloadInfo(
                url=_BASE_URL + "qmnist-train-images-idx3-ubyte.gz",
                checksum="sha256:9e26a7bf1683614e065d7b76460ccd52807165b3f22561fb782bd9f38c52b51d",
            ),
            "train_labels": DownloadInfo(
                url=_BASE_URL + "qmnist-train-labels-idx2-int.gz",
                checksum="sha256:2c05dc77f6b916b38e455e97ab129a42a444f3dbef09b278a366f82904e0dd9f",
            ),
            "test_images": DownloadInfo(
                url=_BASE_URL + "qmnist-test-images-idx3-ubyte.gz",
                checksum="sha256:43fc22bf7498b8fc98de98369d72f752d0deabc280a43a7bcc364ab19e57b375",
            ),
            "test_labels": DownloadInfo(
                url=_BASE_URL + "qmnist-test-labels-idx2-int.gz",
                checksum="sha256:9fbcbe594c3766fdf4f0b15c5165dc0d1e57ac604e01422608bb72c906030d06",
            ),
        },
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "QMNIST reconstructs MNIST from NIST SD19, with 60,000 training "
                "and 60,000 test digits. The first 10,000 test examples match "
                "the original MNIST test identities; the remaining 50,000 "
                "recover its previously unreleased test digits."
            ),
            features=Features(
                {
                    "image": Image(),
                    "label": ClassLabel(names=[str(i) for i in range(10)]),
                    **{name: Value("int32") for name in _METADATA_FIELDS},
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE.homepage,
            citation=self.SOURCE.citation,
            license=self.SOURCE.license,
        )

    def _candidate_splits(self):
        return ["train", "test", "test10k", "test50k"]

    def _split_generators(self):
        assets = self.SOURCE.assets
        paths = {name: download(asset, dest_folder=self._raw_download_dir) for name, asset in assets.items()}
        splits = []
        for split in self._candidate_splits():
            source_split = "train" if split == "train" else "test"
            splits.append(
                SplitGenerator(
                    name=split,
                    gen_kwargs={
                        "images_path": paths[f"{source_split}_images"],
                        "labels_path": paths[f"{source_split}_labels"],
                        "split": split,
                    },
                )
            )
        return splits

    def _generate_examples(self, images_path, labels_path, split):
        if split not in self._candidate_splits():
            raise ValueError(f"Unknown QMNIST split: {split!r}")
        with gzip.open(images_path, "rb") as handle:
            header = handle.read(16)
            if len(header) != 16:
                raise ValueError("Truncated QMNIST image header")
            magic, count, rows, cols = struct.unpack(">IIII", header)
            if magic != 2051 or (rows, cols) != (28, 28):
                raise ValueError("Expected QMNIST IDX unsigned-byte images of shape 28 x 28")
            payload = handle.read()
            if len(payload) != count * rows * cols:
                raise ValueError("QMNIST image payload length does not match its header")
            images = np.frombuffer(payload, dtype=np.uint8).reshape(count, rows, cols)

        with gzip.open(labels_path, "rb") as handle:
            header = handle.read(12)
            if len(header) != 12:
                raise ValueError("Truncated QMNIST label header")
            magic, label_count, columns = struct.unpack(">III", header)
            if magic != 3074 or columns != 8:
                raise ValueError("Expected QMNIST IDX int32 labels with eight columns")
            payload = handle.read()
            if len(payload) != label_count * columns * 4:
                raise ValueError("QMNIST label payload length does not match its header")
            labels = np.frombuffer(payload, dtype=">i4").reshape(label_count, columns)
        if count != label_count:
            raise ValueError("QMNIST image and label counts differ")
        if np.any((labels[:, 0] < 0) | (labels[:, 0] > 9)):
            raise ValueError("QMNIST digit labels must be in [0, 9]")
        if split in ("test10k", "test50k") and count < 10000:
            raise ValueError("QMNIST test subsets require at least 10,000 test rows")

        start = 10000 if split == "test50k" else 0
        stop = 10000 if split == "test10k" else count
        for index in range(start, stop):
            extended_label = labels[index]
            yield (
                index,
                {
                    "image": images[index],
                    "label": int(extended_label[0]),
                    **{name: int(value) for name, value in zip(_METADATA_FIELDS, extended_label[1:])},
                },
            )
