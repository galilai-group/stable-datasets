"""CIFAR-10.2, a reproduction-focused CIFAR-10 distribution-shift benchmark."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from stable_datasets.images._cifar10 import CIFAR10_LABELS, validate_cifar10_arrays
from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Image, Version
from stable_datasets.utils import BaseDatasetBuilder


class CIFAR10_2(BaseDatasetBuilder):
    """CIFAR-10.2 train and test splits for distribution-shift research.

    The release reproduces the CIFAR-10 collection pipeline and provides
    10,000 training examples plus a separate 2,000-example test set.
    """

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://github.com/modestyachts/cifar-10.2",
        assets={
            "train": DownloadInfo(
                url="https://raw.githubusercontent.com/modestyachts/cifar-10.2/master/cifar102_train.npz",
                filename="cifar102_train.npz",
                checksum="sha256:03804f609934c2fdf3adfb708a820e8372284ee5ccac1e4d8aed9f2c87cab099",
            ),
            "test": DownloadInfo(
                url="https://raw.githubusercontent.com/modestyachts/cifar-10.2/master/cifar102_test.npz",
                filename="cifar102_test.npz",
                checksum="sha256:e4fd6462bd5141ed293acd52ea570952f65d1bfd4ec9b068f87742f75a7a632e",
            ),
        },
        citation="""@inproceedings{lu2020harder,
  author={Lu, Shangyun and Nott, Bradley and Olson, Aaron and Todeschini, Alberto and
          Vahabi, Hossein and Carmon, Yair and Schmidt, Ludwig},
  title={Harder or Different? A Closer Look at Distribution Shift in Dataset Reproduction},
  booktitle={ICML Workshop on Uncertainty and Robustness in Deep Learning},
  year={2020},
  url={https://www.gatsby.ucl.ac.uk/~balaji/udl2020/accepted-papers/UDL2020-paper-101.pdf}
}""",
        # The upstream repository does not specify a dataset license.
        license="Unknown",
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "CIFAR-10.2 is a reproduction-focused CIFAR-10 benchmark with 10,000 training and "
                "2,000 test images, all 32x32 RGB images in the CIFAR-10 label space."
            ),
            features=Features({"image": Image(), "label": ClassLabel(names=CIFAR10_LABELS)}),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license=self.SOURCE["license"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path: str | Path, split: str):
        """Yield a CIFAR-10.2 NPZ split while validating its public contract."""
        with np.load(data_path, allow_pickle=False) as data:
            try:
                images = data["images"]
                labels = data["labels"]
            except KeyError as error:
                raise ValueError(f"CIFAR-10.2 {split} archive must contain 'images' and 'labels' arrays.") from error

        validate_cifar10_arrays(images, labels, dataset_name=f"CIFAR-10.2 {split}")

        for index, (image, label) in enumerate(zip(images, labels, strict=True)):
            yield index, {"image": image, "label": int(label)}
