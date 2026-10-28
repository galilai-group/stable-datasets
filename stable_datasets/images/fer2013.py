import csv
import hashlib
import os
import tarfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, _default_dest_folder


_KAGGLE_URL = "https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge"

# Kaggle-hosted files accepted by the builder, with the MD5s published by torchvision. Both CSVs hold the same
# 35,887 faces; icml_face_data.csv uses a different column order and spaces in its header (`emotion, Usage, pixels`).
_CSV_MD5 = {
    "fer2013.csv": "f8428a1edbd21e88f42c73edd2a14f95",
    "icml_face_data.csv": "b114b9e04e6949e5fe8b6a98b3892b1d",
}
_TARBALL = "fer2013.tar.gz"  # Kaggle archive that contains fer2013/fer2013.csv

# FER2013 `Usage` values -> splits (the same mapping as the FER+ release).
_USAGE_TO_SPLIT = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}


class FER2013(BaseDatasetBuilder):
    """FER2013 (Facial Expression Recognition 2013).

    Abstract
    48x48 grayscale face crops labelled with one of seven facial expressions: angry, disgust, fear, happy, sad,
    surprise and neutral. Released for the ICML 2013 "Challenges in Representation Learning" Kaggle competition, it
    is one of the 27 datasets of CLIP's linear-probe evaluation suite.

    Context
    The faces were collected with Google image searches for emotion-related keywords, then automatically registered
    and cropped, so pose, lighting, age and occlusion vary widely. Human accuracy is estimated at ~65%, and the classes
    are imbalanced (disgust has 547 examples, happy 8,989), which makes it a demanding transfer benchmark.

    Content
    - Images: 35,887 grayscale 48x48 faces, decoded from the space-separated `pixels` column of the competition CSV.
    - Labels: the `emotion` column, 0-6 in the order angry, disgust, fear, happy, sad, surprise, neutral.
    - Splits: the competition's `Usage` column. Training -> train (28,709), PublicTest -> validation (3,589),
      PrivateTest -> test (3,589).

    Download
    Kaggle requires an account and acceptance of the competition rules, so the data cannot be fetched
    automatically. Download `fer2013.csv` (or `fer2013.tar.gz` / `icml_face_data.csv`) from the competition page and
    place it in the builder's download directory (`download_dir`, default `~/.stable-datasets/downloads`).
    """

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage=_KAGGLE_URL,
        assets={},  # manual download, see the class docstring
        license="Kaggle competition rules",
        citation="""@inproceedings{goodfellow2013challenges,
                    title={Challenges in Representation Learning: A report on three machine learning contests},
                    author={Goodfellow, Ian J. and Erhan, Dumitru and Carrier, Pierre Luc and Courville, Aaron and
                            Mirza, Mehdi and Hamner, Ben and Cukierski, Will and Tang, Yichuan and Thaler, David and
                            Lee, Dong-Hyun and others},
                    booktitle={International Conference on Neural Information Processing (ICONIP)},
                    year={2013}}""",
    )

    def _info(self):
        return DatasetInfo(
            description="""FER2013: 48x48 grayscale face images labelled with one of seven facial expressions
                (angry, disgust, fear, happy, sad, surprise, neutral), from the ICML 2013 Challenges in Representation
                Learning Kaggle competition.""",
            features=Features({"image": ImageFeature(), "label": ClassLabel(names=self._labels())}),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license=self.SOURCE["license"],
            citation=self.SOURCE["citation"],
        )

    def _candidate_splits(self) -> list:
        return [Split.TRAIN, Split.VALIDATION, Split.TEST]

    def _split_generators(self):
        csv_path = _find_csv(Path(getattr(self, "_raw_download_dir", None) or _default_dest_folder()))
        return [
            SplitGenerator(name=name, gen_kwargs={"csv_path": csv_path, "split": key})
            for name, key in ((Split.TRAIN, "train"), (Split.VALIDATION, "val"), (Split.TEST, "test"))
        ]

    def _generate_examples(self, csv_path, split):
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            columns = {name.strip(): i for i, name in enumerate(next(reader))}
            for idx, row in enumerate(tqdm(reader, desc=f"Processing FER2013 {split} split")):
                if _USAGE_TO_SPLIT[row[columns["Usage"]].strip()] != split:
                    continue
                pixels = np.array(row[columns["pixels"]].split(), dtype=np.uint8).reshape(48, 48)
                yield f"{idx:05d}", {"image": pixels, "label": int(row[columns["emotion"]])}

    @staticmethod
    def _labels():
        return ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_csv(download_dir: Path) -> Path:
    """Locate (and checksum) the Kaggle FER2013 CSV in ``download_dir``, extracting the tarball if needed."""
    tarball = download_dir / _TARBALL
    if not (download_dir / "fer2013.csv").exists() and tarball.exists():
        with tarfile.open(tarball, "r:gz") as archive:
            member = next(m for m in archive.getmembers() if m.name.endswith("fer2013.csv"))
            with open(download_dir / "fer2013.csv", "wb") as out:
                out.write(archive.extractfile(member).read())

    for name, expected in _CSV_MD5.items():
        path = download_dir / name
        if not path.exists():
            continue
        if _md5(path) != expected:
            raise RuntimeError(f"{path} has an unexpected MD5 checksum; re-download it from {_KAGGLE_URL}")
        return path

    raise FileNotFoundError(
        f"FER2013 must be downloaded manually (Kaggle login and rule acceptance required). Download "
        f"'fer2013.csv' (or '{_TARBALL}' / 'icml_face_data.csv') from {_KAGGLE_URL}/data and place it in "
        f"{download_dir}{os.sep}"
    )
