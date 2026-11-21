"""CMU Face Images with a deterministic subject-disjoint split."""

import io
import tarfile
import zipfile
from pathlib import PurePosixPath

from stable_datasets.schema import (
    ClassLabel,
    DatasetInfo,
    DatasetSource,
    DownloadInfo,
    Features,
    Value,
    Version,
)
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


_URL = "https://archive.ics.uci.edu/static/public/124/cmu+face+images.zip"
_SHA256 = "01a8b6cc58ac2f825f72abd6cf2b393df067c6529d96f7a90b22dede71b142cc"

# Fixed identities make the custom split independent of archive ordering.
_SUBJECTS = (
    "an2i",
    "at33",
    "boland",
    "bpm",
    "ch4f",
    "cheyer",
    "choon",
    "danieln",
    "glickman",
    "karyadi",
    "kawamura",
    "kk49",
    "megak",
    "mitchell",
    "night",
    "phoebe",
    "saavik",
    "steffi",
    "sz24",
    "tammo",
)

_SPLIT_SUBJECTS = {
    "train": frozenset(_SUBJECTS[:12]),
    "validation": frozenset(_SUBJECTS[12:16]),
    "test": frozenset(_SUBJECTS[16:]),
}

_POSES = ("straight", "left", "right", "up")
_EXPRESSIONS = ("neutral", "happy", "sad", "angry")
_EYES = ("open", "sunglasses")


class CMUFaces(BaseDatasetBuilder):
    """Four-class head-pose classification using 32-by-30 grayscale images.

    This builder defines a custom 12/4/4-person train/validation/test split.
    It is not an official benchmark split. Other image resolutions are
    excluded so that resized copies cannot enter different splits.
    """

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://archive.ics.uci.edu/dataset/124/cmu+face+images",
        assets={
            "train": DownloadInfo(url=_URL, checksum=f"sha256:{_SHA256}"),
            "validation": DownloadInfo(url=_URL, checksum=f"sha256:{_SHA256}"),
            "test": DownloadInfo(url=_URL, checksum=f"sha256:{_SHA256}"),
        },
        license="CC BY 4.0",
        citation="""@misc{cmu_face_images_124,
    author = {Mitchell, Tom},
    title = {CMU Face Images},
    publisher = {UCI Machine Learning Repository},
    year = {1997},
    doi = {10.24432/C5JC79},
    url = {https://archive.ics.uci.edu/dataset/124/cmu+face+images}
}""",
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "CMU Face Images contains grayscale face photographs of "
                "20 people with variations in head pose, expression, and "
                "eye condition. This builder uses the 624 valid PGM files "
                "in faces_4.tar.gz at width 32 and height 30. The label is "
                "head pose. A custom subject-disjoint split assigns "
                "12 people to training, four to validation, and four to "
                "testing. Files marked as bad by the source are excluded."
            ),
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=list(_POSES)),
                    "subject": Value("string"),
                    "expression": ClassLabel(names=list(_EXPRESSIONS)),
                    "eyes": ClassLabel(names=list(_EYES)),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license=self.SOURCE["license"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        if split not in _SPLIT_SUBJECTS:
            raise ValueError(f"Unknown CMUFaces split: {split!r}")

        subjects = _SPLIT_SUBJECTS[split]

        with zipfile.ZipFile(data_path) as archive:
            tar_bytes = archive.read("faces_4.tar.gz")

        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as archive:
            for member in sorted(archive.getmembers(), key=lambda item: item.name):
                if not member.isfile() or not member.name.endswith(".pgm"):
                    continue

                filename = PurePosixPath(member.name)
                parts = filename.stem.split("_")

                if len(parts) != 5:
                    raise ValueError(f"Unexpected CMU filename: {member.name}")

                subject, pose, expression, eyes, scale = parts

                if subject not in _SUBJECTS or scale != "4":
                    raise ValueError(f"Unexpected subject or scale: {member.name}")

                if pose not in _POSES or expression not in _EXPRESSIONS or eyes not in _EYES:
                    raise ValueError(f"Unexpected CMU annotation: {member.name}")

                if subject not in subjects:
                    continue

                image_file = archive.extractfile(member)
                if image_file is None:
                    raise ValueError(f"Cannot read CMU image: {member.name}")

                with image_file:
                    image_bytes = image_file.read()

                yield (
                    member.name,
                    {
                        "image": image_bytes,
                        "label": self.info.features["label"].str2int(pose),
                        "subject": subject,
                        "expression": self.info.features["expression"].str2int(expression),
                        "eyes": self.info.features["eyes"].str2int(eyes),
                    },
                )
