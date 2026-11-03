import csv
import io
import zipfile

from stable_datasets.schema import (
    BuilderConfig,
    ClassLabel,
    DatasetInfo,
    DatasetSource,
    DownloadInfo,
    Features,
    Version,
)
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, bulk_download


# The authors' repository (labels.csv + the 15 official split CSVs) as one zip, pinned to the verified commit.
# Downloaded as an archive because raw.githubusercontent.com serves the CSVs gzip-compressed, which
# the library's download size check rejects (same issue as PR #72).
_REPO_ZIP_URL = "https://codeload.github.com/AlexOlsen/DeepWeeds/zip/da084f62bcd1a2b0afeb8b81f1a27be3186391a1"


def _read_csv(repo_zip, name):
    """Read 'labels/<name>' from the repository zip, whose top folder is 'DeepWeeds-<commit>/'."""
    member = next(n for n in repo_zip.namelist() if n.endswith(f"/labels/{name}"))
    with repo_zip.open(member) as f:
        return list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", newline="")))


class DeepWeeds(BaseDatasetBuilder):
    """DeepWeeds dataset for classification of eight classes of weeds as well as a negative class in the Australian rangelands."""

    VERSION = Version("1.0.0")
    BUILDER_CONFIGS = [
        BuilderConfig(name=f"subset{k}", description=f"Official train/val/test split {k} of 5") for k in range(5)
    ]

    # Single source-of-truth for dataset provenance + download locations.
    SOURCE = DatasetSource(
        homepage="https://github.com/AlexOlsen/DeepWeeds",
        assets={
            "images": DownloadInfo(
                url="https://drive.usercontent.google.com/download?id=1xnK3B6K6KekDI55vwJ0vnc2IGoDga9cj&export=download&confirm=t",
                filename="deepweeds_images.zip",
            ),
            "labels": DownloadInfo(url=_REPO_ZIP_URL, filename="deepweeds_labels.zip"),
        },
        citation="""@article{DeepWeeds2019,
                        author = {Alex Olsen and
                            Dmitry A. Konovalov and
                            Bronson Philippa and
                            Peter Ridd and
                            Jake C. Wood and
                            Jamie Johns and
                            Wesley Banks and
                            Benjamin Girgenti and
                            Owen Kenny and
                            James Whinney and
                            Brendan Calvert and
                            Mostafa {Rahimi Azghadi} and
                            Ronald D. White},
                        title = {{DeepWeeds: A Multiclass Weed Species Image Dataset for Deep Learning}},
                        journal = {Scientific Reports},
                        year = 2019,
                        number = 2058,
                        month = 2,
                        volume = 9,
                        issue = 1,
                        day = 14,
                        url = "https://doi.org/10.1038/s41598-018-38343-3",
                        doi = "10.1038/s41598-018-38343-3"
                        }
                        """,
    )

    def _info(self):
        return DatasetInfo(
            description="""
                            DeepWeeds is an image classification dataset containing 17,509 in-situ images
                            collected across several locations in Queensland, Australia. The dataset includes
                            eight invasive weed species—Chinee apple, Lantana, Parkinsonia, Parthenium,
                            Prickly acacia, Rubber vine, Siam weed, and Snake weed as well as a negative
                            class containing images without the target weed species. Images were captured
                            under natural field conditions with surrounding vegetation, providing a
                            realistic benchmark for weed species recognition and agricultural computer vision.
                        """,
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(
                        names=[
                            "Chinee_apple",
                            "Lantana",
                            "Parkinsonia",
                            "Parthenium",
                            "Prickly_acacia",
                            "Rubber_vine",
                            "Siam_weed",
                            "Snake_weed",
                            "Negative",
                        ]
                    ),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license="CC BY 4.0",
            citation=self.SOURCE["citation"],
        )

    def _split_generators(self):
        # Step 1: Download every file in the SOURCE assets dictionary (skips files already on disk).
        assets = self._source()["assets"]
        names = list(assets)
        files = bulk_download(
            [self._normalize_download_info(assets[name], asset_name=name) for name in names],
            dest_folder=self._raw_download_dir,
        )
        paths = dict(zip(names, files))  # e.g. paths["images"] = local path of the zip file

        # Step 2: Choice of the 5 official train/val/test splits. Each split has its own CSV file for train, val, and test.
        subset = self.config.name  # e.g. subset = "subset0" for the first split

        # Step 3: One order per split: same images + labels, different split file.
        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "images_path": paths["images"],
                    "labels_path": paths["labels"],
                    "split_file": f"train_{subset}.csv",
                },
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={
                    "images_path": paths["images"],
                    "labels_path": paths["labels"],
                    "split_file": f"val_{subset}.csv",
                },
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={
                    "images_path": paths["images"],
                    "labels_path": paths["labels"],
                    "split_file": f"test_{subset}.csv",
                },
            ),
        ]

    def _generate_examples(self, images_path, labels_path, split_file):
        with zipfile.ZipFile(labels_path) as repo:
            labels = {row["Filename"]: int(row["Label"]) for row in _read_csv(repo, "labels.csv")}
            filenames = [row["Filename"] for row in _read_csv(repo, split_file)]
        with zipfile.ZipFile(images_path) as archive:
            for filename in filenames:
                yield (
                    filename,
                    {
                        "image": archive.read(filename),
                        "label": labels[filename],
                    },
                )

    def _candidate_splits(self):
        return [Split.TRAIN, Split.VALIDATION, Split.TEST]
