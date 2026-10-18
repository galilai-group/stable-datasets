"""Official German Traffic Sign Recognition Benchmark (GTSRB)."""

import csv
import io
import zipfile
from pathlib import PurePosixPath

from PIL import Image as PILImage

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Image, Version
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, bulk_download


_ARCHIVE_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370"
_TRAIN_PREFIX = "GTSRB/Final_Training/Images/"
_TEST_PREFIX = "GTSRB/Final_Test/Images/"
_TEST_GROUND_TRUTH = "GT-final_test.csv"


class GTSRB(BaseDatasetBuilder):
    """GTSRB classification with the official final train and labeled test splits."""

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://github.com/houbensebastian/GermanTrafficSignBenchmarks",
        assets={
            "train": DownloadInfo(url=f"{_ARCHIVE_URL}/GTSRB_Final_Training_Images.zip"),
            "test_images": DownloadInfo(url=f"{_ARCHIVE_URL}/GTSRB_Final_Test_Images.zip"),
            "test_ground_truth": DownloadInfo(url=f"{_ARCHIVE_URL}/GTSRB_Final_Test_GT.zip"),
        },
        citation=r"""@inproceedings{Stallkamp-IJCNN-2011,
    author = {Johannes Stallkamp and Marc Schlipsing and Jan Salmen and Christian Igel},
    title = {The {G}erman {T}raffic {S}ign {R}ecognition {B}enchmark: A multi-class classification competition},
    booktitle = {IEEE International Joint Conference on Neural Networks},
    pages = {1453--1460},
    year = {2011}
}""",
    )

    def _info(self):
        return DatasetInfo(
            description="Official GTSRB final training and test images with 43 traffic sign classes.",
            features=Features({"image": Image(), "label": ClassLabel(num_classes=43)}),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE.homepage,
            citation=self.SOURCE.citation,
            license=self.SOURCE.license,
        )

    def _candidate_splits(self):
        return [Split.TRAIN, Split.TEST]

    def _split_generators(self):
        assets = self._source().assets
        keys = ("train", "test_images", "test_ground_truth")
        paths = bulk_download([assets[key] for key in keys], dest_folder=self._raw_download_dir)
        by_key = dict(zip(keys, paths))
        return [
            SplitGenerator(name=Split.TRAIN, gen_kwargs={"data_path": by_key["train"], "split": Split.TRAIN}),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={
                    "data_path": by_key["test_images"],
                    "ground_truth_path": by_key["test_ground_truth"],
                    "split": Split.TEST,
                },
            ),
        ]

    def _generate_examples(self, data_path, split, ground_truth_path=None):
        if split == Split.TRAIN:
            prefix = _TRAIN_PREFIX
        else:
            prefix = _TEST_PREFIX
            # Test labels are stored separately from the images.
            with zipfile.ZipFile(ground_truth_path) as archive:
                with archive.open(_TEST_GROUND_TRUTH) as raw, io.TextIOWrapper(raw, encoding="utf-8-sig") as text:
                    labels = {row["Filename"]: int(row["ClassId"]) for row in csv.DictReader(text, delimiter=";")}

        with zipfile.ZipFile(data_path) as archive:
            # Keep row order independent of ZIP entry order.
            for member in sorted(archive.infolist(), key=lambda item: item.filename):
                if not member.filename.startswith(prefix) or not member.filename.endswith(".ppm"):
                    continue
                filename = PurePosixPath(member.filename)
                label = int(filename.parent.name) if split == Split.TRAIN else labels[filename.name]
                with archive.open(member) as raw, PILImage.open(raw) as source_image:
                    image = source_image.convert("RGB")
                yield member.filename, {"image": image, "label": label}
