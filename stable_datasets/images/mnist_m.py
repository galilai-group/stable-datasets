import tarfile
from io import BytesIO

import torch
from PIL import Image as PILImage
from tqdm import tqdm

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


_RELEASE_URL = "https://github.com/liyxi/mnist-m/releases/download/data"


class MNISTM(BaseDatasetBuilder):
    """MNIST-M dataset for digit classification and domain adaptation.

    MNIST-M was introduced by Ganin & Lempitsky (2015) as the target domain for unsupervised domain adaptation from
    MNIST. Each image is an MNIST digit blended over a random color patch cropped from the BSDS500 photo collection,
    using the absolute difference between the digit and the patch. The result keeps MNIST's labels and 28x28 layout
    but has natural-image color and texture statistics, so models trained only on MNIST transfer poorly to it.

    The dataset has 60000 training and 10000 test RGB images of size 28x28 in 10 classes (digits 0-9). The data
    originates from the DANN paper's generation procedure (see ``VanushVaswani/keras_mnistm``). This builder reads
    the labeled release mirrored at https://github.com/liyxi/mnist-m, where each split is a ``torch.save``'d
    ``(images, labels)`` tuple of a ``uint8`` tensor of shape ``(N, 28, 28, 3)`` and an ``int64`` tensor of shape
    ``(N,)``.
    """

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://github.com/liyxi/mnist-m",
        assets={
            # MD5 (as published by the mirror): 191ed53db9933bd85cc9700558847391
            "train": DownloadInfo(
                url=f"{_RELEASE_URL}/mnist_m_train.pt.tar.gz",
                checksum="sha256:69c983f2d2f9b0280efc691f8252576b18bf6e5dee5279dbf43109a023d151de",
            ),
            # MD5 (as published by the mirror): e11cb4d7fff76d7ec588b1134907db59
            "test": DownloadInfo(
                url=f"{_RELEASE_URL}/mnist_m_test.pt.tar.gz",
                checksum="sha256:85958398c49815ac2f38aa309eb6d7d3d54c5bf9e0e3946cb00259b58a8b7cf7",
            ),
        },
        citation="""@inproceedings{ganin2015unsupervised,
                    title     = {Unsupervised Domain Adaptation by Backpropagation},
                    author    = {Ganin, Yaroslav and Lempitsky, Victor},
                    booktitle = {Proceedings of the 32nd International Conference on Machine Learning},
                    pages     = {1180--1189},
                    year      = {2015}}""",
        license="Unknown",
    )

    def _info(self):
        return DatasetInfo(
            description="""MNIST-M blends MNIST digits over random color patches from BSDS500. It contains
                           60000 training and 10000 test 28x28 RGB images in 10 classes, and is a standard
                           target domain for MNIST -> MNIST-M domain adaptation.""",
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=[str(i) for i in range(10)]),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
            license=self.SOURCE["license"],
        )

    def _generate_examples(self, data_path, split):
        images, labels = self._load_tensors(data_path, split)
        images, labels = images.numpy(), labels.numpy()

        for idx in tqdm(range(len(labels)), desc=f"Processing {split} split"):
            yield idx, {"image": PILImage.fromarray(images[idx]), "label": int(labels[idx])}

    @staticmethod
    def _load_tensors(data_path, split):
        """Load the ``(images, labels)`` tensors from ``mnist_m_<split>.pt`` inside the tar.gz."""
        with tarfile.open(data_path, "r:gz") as tar:
            # The release uses torch's legacy serialization format, which needs a seekable file.
            buffer = BytesIO(tar.extractfile(f"mnist_m_{split}.pt").read())
        # weights_only restricts unpickling to tensors and primitive containers.
        images, labels = torch.load(buffer, weights_only=True)

        if images.dtype != torch.uint8 or images.ndim != 4 or tuple(images.shape[1:]) != (28, 28, 3):
            raise ValueError(
                f"Expected uint8 images of shape (N, 28, 28, 3), got {images.dtype} {tuple(images.shape)}."
            )
        if labels.ndim != 1 or len(labels) != len(images):
            raise ValueError(f"Expected {len(images)} labels, got shape {tuple(labels.shape)}.")
        return images, labels
