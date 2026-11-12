import tarfile
from io import BytesIO
from pathlib import PurePosixPath

from PIL import Image as PILImage
from tqdm import tqdm

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


# Original release by the authors, linked as "MNIST-M" from https://yaroslav.ganin.net/ (Google Drive file
# 0B_tExHiYS-0veklUZHFYT19KYjg). ``confirm=t`` skips the virus-scan page Drive shows for large files.
_URL = (
    "https://drive.usercontent.google.com/download?id=0B_tExHiYS-0veklUZHFYT19KYjg"
    "&export=download&confirm=t&resourcekey=0-DE7ivVVSn8PwP9fdHvSUfA"
)
# No checksum is published with the release; this is the SHA-256 of mnist_m.tar.gz as served by the link above
# (MD5 859df31c91afe82e80e5012ba928f279).
_CHECKSUM = "sha256:022d6584383f38fce6f53ca1387dcac9a51986cee5e4d231544546ec103501a6"

IMAGE_SIZE = (32, 32)


class MNISTM(BaseDatasetBuilder):
    """MNIST-M dataset for digit classification and domain adaptation.

    MNIST-M was introduced by Ganin & Lempitsky (2015) as the target domain for unsupervised domain adaptation from
    MNIST. Each image is an MNIST digit blended over a random color patch cropped from the BSDS500 photo collection,
    using the absolute difference between the digit and the patch. The result keeps MNIST's labels but has
    natural-image color and texture statistics, so models trained only on MNIST transfer poorly to it.

    This builder reads the original release by the authors (``mnist_m.tar.gz``, linked from Yaroslav Ganin's
    homepage). It has 59001 training and 9001 test RGB images of size 32x32 in 10 classes (digits 0-9), stored as
    PNG files with one ``<file name> <label>`` labels file per split. Other copies circulating online, such as
    60000/10000-image regenerations at 28x28, are not the same data.
    """

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://yaroslav.ganin.net/",
        assets={
            "train": DownloadInfo(url=_URL, checksum=_CHECKSUM, filename="mnist_m.tar.gz"),
            "test": DownloadInfo(url=_URL, checksum=_CHECKSUM, filename="mnist_m.tar.gz"),
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
            description="""MNIST-M blends MNIST digits over random color patches from BSDS500. The original
                           release contains 59001 training and 9001 test 32x32 RGB images in 10 classes, and is a
                           standard target domain for MNIST -> MNIST-M domain adaptation.""",
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
        labels, images = self._read_split(data_path, split)

        for idx, (file_name, label) in enumerate(tqdm(labels, desc=f"Processing {split} split")):
            image_bytes = images[file_name]
            image = PILImage.open(BytesIO(image_bytes))
            if image.mode != "RGB" or image.size != IMAGE_SIZE:
                raise ValueError(f"Expected a 32x32 RGB image for {file_name}, got {image.mode} {image.size}.")
            # The source PNGs are stored as-is; re-encoding would not change the decoded pixels.
            yield idx, {"image": image_bytes, "label": label}

    @staticmethod
    def _read_split(data_path, split):
        """Read ``[(file_name, label)]`` from the split's labels file and the split's PNG bytes keyed by file name."""
        labels_name = f"mnist_m_{split}_labels.txt"
        image_dir = f"mnist_m_{split}"
        labels_text, images = None, {}

        with tarfile.open(data_path, "r|gz") as tar:
            for member in tar:
                parts = PurePosixPath(member.name).parts
                if not member.isfile() or parts[0] != "mnist_m" or parts[-1].startswith("."):
                    continue
                if parts[1:] == (labels_name,):
                    labels_text = tar.extractfile(member).read().decode()
                elif len(parts) == 3 and parts[1] == image_dir and parts[2].endswith(".png"):
                    images[parts[2]] = tar.extractfile(member).read()

        if labels_text is None:
            raise ValueError(f"mnist_m/{labels_name} not found in {data_path}.")

        labels = []
        for line in labels_text.splitlines():
            if not line.strip():
                continue
            file_name, label = line.split()
            if not label.isdigit() or not 0 <= int(label) < 10:
                raise ValueError(f"Invalid label {label!r} for {file_name} in {labels_name}.")
            labels.append((file_name, int(label)))

        labeled = {file_name for file_name, _ in labels}
        if len(labeled) != len(labels) or labeled != set(images):
            raise ValueError(
                f"{labels_name} lists {len(labels)} entries ({len(labeled)} unique) but mnist_m/{image_dir}/ has "
                f"{len(images)} PNG files; they must match one to one."
            )
        return labels, images
