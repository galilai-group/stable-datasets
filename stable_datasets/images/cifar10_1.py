"""CIFAR-10.1, a fresh test set for measuring CIFAR-10 generalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from stable_datasets.images._cifar10 import CIFAR10_LABELS, validate_cifar10_arrays
from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Image, Version
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, bulk_download


class CIFAR10_1(BaseDatasetBuilder):
    """The class-balanced v6 release of the CIFAR-10.1 test set.

    CIFAR-10.1 contains 2,000 independently collected 32x32 RGB images. It
    uses the original CIFAR-10 label space and is intended for evaluating a
    model trained on CIFAR-10; it has no training split.
    """

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://github.com/modestyachts/CIFAR-10.1",
        assets={
            "images": DownloadInfo(
                url="https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/master/datasets/cifar10.1_v6_data.npy",
                filename="cifar10.1_v6_data.npy",
                checksum="sha256:2997188e5816f5bd545dc77771b6227828c28146049fcecf3fa10775474cacc6",
            ),
            "labels": DownloadInfo(
                url="https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/master/datasets/cifar10.1_v6_labels.npy",
                filename="cifar10.1_v6_labels.npy",
                checksum="sha256:ae40beda001693674edc94d925ee8268cfe68905f8f9aff800c8dcdfcd6c9448",
            ),
        },
        citation="""@article{recht2018cifar10,
  author={Recht, Benjamin and Roelofs, Rebecca and Schmidt, Ludwig and Shankar, Vaishaal},
  title={Do CIFAR-10 Classifiers Generalize to CIFAR-10?},
  journal={arXiv preprint arXiv:1806.00451},
  year={2018}
}""",
        # The upstream MIT license excludes the image and label data.
        license="Unknown",
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "CIFAR-10.1 v6 is a class-balanced, independently collected test set of 2,000 "
                "32x32 RGB images in the CIFAR-10 label space."
            ),
            features=Features({"image": Image(), "label": ClassLabel(names=CIFAR10_LABELS)}),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license=self.SOURCE["license"],
            citation=self.SOURCE["citation"],
        )

    def _candidate_splits(self) -> list:
        """CIFAR-10.1 has two raw assets but one logical test split."""
        return [Split.TEST]

    def _split_generators(self):
        source = self._source()
        assets = source["assets"]
        paths = bulk_download(
            [assets["images"], assets["labels"]],
            dest_folder=self._raw_download_dir,
        )
        return [
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={"images_path": paths[0], "labels_path": paths[1]},
            )
        ]

    def _generate_examples(self, images_path: str | Path, labels_path: str | Path):
        """Yield the paired image and label arrays stored in the v6 release."""
        images = np.load(images_path, allow_pickle=False)
        labels = np.load(labels_path, allow_pickle=False)
        validate_cifar10_arrays(images, labels, dataset_name="CIFAR-10.1")

        for index, (image, label) in enumerate(zip(images, labels, strict=True)):
            yield index, {"image": image, "label": int(label)}
