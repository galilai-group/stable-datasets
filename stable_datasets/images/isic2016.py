"""ISIC 2016 Task 3 original image classification data and official labels."""

import csv
import re
import zipfile
from pathlib import Path, PurePosixPath

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
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, bulk_download


_BASE_URL = "https://isic-archive.s3.amazonaws.com/challenges/2016/"
_ASSET_HASHES = {
    "Training_Data.zip": "d78bd1de08511b514e58784c91fd2d8dab24e407cafd023a7debbceb9155ffb2",
    "Training_GroundTruth.csv": "0ffadd63b7cec650cddfbe5df54d1d3167453fe6004c2bbf05824184b0c4d405",
    "Test_Data.zip": "7c52d5c013b6a79b92c246d48d9f175299670537c128bcc1400b35891f6350e6",
    "Test_GroundTruth.csv": "8f7718bfe5ed110e0235b0d9afc40fe89e52610c5687cc9cd7dd0c23352b82ce",
}


class ISIC2016(BaseDatasetBuilder):
    """ISIC 2016 Task 3 dermoscopic skin lesion classification dataset.

    Contains 900 training and 379 test JPEG images with variable dimensions.
    Labels identify benign lesions (0) and malignant lesions (1). Images and
    labels were released for the ISBI 2016 skin lesion analysis challenge;
    the official train/test partitions are retained.
    """

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="https://challenge.isic-archive.com/landing/2016/39/",
        citation="""@article{gutman2016skin,
            title={Skin Lesion Analysis toward Melanoma Detection: A Challenge at the International
                   Symposium on Biomedical Imaging (ISBI) 2016, hosted by the International Skin
                   Imaging Collaboration (ISIC)},
            author={Gutman, David and Codella, Noel C. F. and Celebi, Emre and Helba, Brian
                    and Marchetti, Michael and Mishra, Nabin and Halpern, Allan},
            journal={arXiv preprint arXiv:1605.01397}, year={2016}}
        """,
        assets={
            suffix: DownloadInfo(url=f"{_BASE_URL}ISBI2016_ISIC_Part3_{suffix}", checksum=f"sha256:{digest}")
            for suffix, digest in _ASSET_HASHES.items()
        },
        license="CC0-1.0",
    )

    def _info(self):
        return DatasetInfo(
            description="ISIC 2016 Task 3: 900 training and 379 test dermoscopic images for lesion classification.",
            features=Features(
                {"image": Image(), "label": ClassLabel(names=["benign", "malignant"]), "image_id": Value("string")}
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE.homepage,
            citation=self.SOURCE.citation,
            license=self.SOURCE.license,
        )

    def _candidate_splits(self):
        return [Split.TRAIN, Split.TEST]

    def _split_generators(self):
        assets = self.SOURCE.assets
        downloaded = bulk_download(assets.values(), dest_folder=self._raw_download_dir)
        paths = dict(zip(assets, downloaded))
        return [
            SplitGenerator(
                split,
                {
                    "data_path": paths[f"{prefix}_Data.zip"],
                    "labels_path": paths[f"{prefix}_GroundTruth.csv"],
                    "split": split,
                },
            )
            for split, prefix in ((Split.TRAIN, "Training"), (Split.TEST, "Test"))
        ]

    @staticmethod
    def _read_labels(labels_path, split):
        encoding = {"benign": 0, "malignant": 1} if split == Split.TRAIN else {"0.0": 0, "1.0": 1}
        labels = {}
        with Path(labels_path).open(newline="", encoding="utf-8-sig") as stream:
            for line, row in enumerate(csv.reader(stream), start=1):
                if len(row) != 2:
                    raise ValueError(f"ISIC2016 label row {line}: expected exactly two columns, without a header.")
                image_id, value = (cell.strip() for cell in row)
                if not re.fullmatch(r"ISIC_\d{7}", image_id) or value not in encoding:
                    raise ValueError(f"Invalid ISIC2016 {split} label row {line}: {row!r}")
                if image_id in labels:
                    raise ValueError(f"Duplicate ISIC2016 label ID: {image_id}")
                labels[image_id] = encoding[value]
        if not labels:
            raise ValueError("ISIC2016 label file is empty.")
        return labels

    def _generate_examples(self, data_path, labels_path, split):
        if split not in (Split.TRAIN, Split.TEST):
            raise ValueError(f"Unknown ISIC2016 split: {split}")
        labels = self._read_labels(labels_path, split)
        with zipfile.ZipFile(data_path) as archive:
            images = {}
            for member in archive.infolist():
                path = PurePosixPath(member.filename)
                if member.is_dir() or "__MACOSX" in path.parts or path.suffix.lower() != ".jpg":
                    continue
                image_id = path.stem
                if image_id in images:
                    raise ValueError(f"Duplicate ISIC2016 image ID: {image_id}")
                images[image_id] = member
            if images.keys() != labels.keys():
                missing = sorted(labels.keys() - images.keys())
                unlabelled = sorted(images.keys() - labels.keys())
                raise ValueError(
                    f"ISIC2016 image/label mismatch: missing images={missing}, unlabelled images={unlabelled}"
                )
            for image_id, member in sorted(images.items()):
                yield image_id, {"image": archive.read(member), "label": labels[image_id], "image_id": image_id}
