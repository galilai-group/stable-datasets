import csv
import io
import random
import zipfile
from collections import defaultdict
from pathlib import Path

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Value, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder, download


KAGGLE_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/andrewmvd/"
    "lung-and-colon-cancer-histopathological-images?datasetVersionNumber=1"
)

# LC25000-clean maps every image to the original tile it was augmented from.
GROUPS_INFO = DownloadInfo(
    url="https://www.kaggle.com/api/v1/datasets/download/gbatchkala/lc25000-clean-groups?datasetVersionNumber=3",
    filename="lc25000_clean_groups.zip",
)
GROUPS_FILE = "lc25000_image_groups.csv"

TEST_FRACTION = 0.2
SPLIT_SEED = 0


def split_groups(group_rows):
    """Map each group id to "train" or "test", holding out 20% of the groups of every class.
    All copies of one original tile share a group, so no tile appears in both splits."""
    groups_per_label = defaultdict(set)
    for row in group_rows:
        groups_per_label[row["label"]].add(int(row["group_id"]))

    group_split = {}
    for label in sorted(groups_per_label):
        groups = sorted(groups_per_label[label])
        random.Random(SPLIT_SEED).shuffle(groups)
        test_count = max(1, round(len(groups) * TEST_FRACTION))
        for index, group_id in enumerate(groups):
            group_split[group_id] = "test" if index < test_count else "train"
    return group_split


class LC25000(BaseDatasetBuilder):
    """Lung and Colon Cancer Histopathological Image Dataset (LC25000)

    This dataset has 25,000 color histopathology images of 768x768 pixels in five balanced
    classes of 5,000: colon adenocarcinoma, benign colon, lung adenocarcinoma, benign lung
    and lung squamous cell carcinoma. The images were made by rotating and flipping 1,250
    original tiles.

    There is no official split, so 20% of the LC25000-clean tile groups of each class are
    held out for test with a fixed seed. No tile is in both splits.
    """

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://github.com/tampapath/lung_colon_image_set",
        assets={
            "train": DownloadInfo(url=KAGGLE_URL, filename="lc25000.zip"),
            "test": DownloadInfo(url=KAGGLE_URL, filename="lc25000.zip"),
        },
        citation="""@article{borkowski2019lc25000,
                      title={Lung and Colon Cancer Histopathological Image Dataset (LC25000)},
                      author={Borkowski, Andrew A and Bui, Marilyn M and Thomas, L Brannon and Wilson, Catherine P and DeLand, Lauren A and Mastorides, Stephen M},
                      journal={arXiv preprint arXiv:1912.12142},
                      year={2019}}
                    @inproceedings{batchkala2024lc25000clean,
                      title={Evaluating Histopathology Foundation Models for Few-Shot Tissue Clustering: An Application to LC25000 Augmented Dataset Cleaning},
                      author={Batchkala, George and Li, Bin and Rittscher, Jens},
                      booktitle={Data Engineering in Medical Imaging (DEMI), MICCAI Workshop},
                      year={2024}}""",
        license="CC BY-SA 4.0 (images), CC BY 4.0 (LC25000-clean groups)",
    )

    def _info(self):
        return DatasetInfo(
            description="Lung and colon histopathology images in five classes, split by original tile.",
            features=Features(
                {
                    "image": ImageFeature(encode_format="JPEG"),
                    "label": ClassLabel(names=self._labels()),
                    "group_id": Value("int32"),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license=self.SOURCE["license"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        """Read the images of one split from the zip. The folder name is the label."""
        groups_path = download(GROUPS_INFO, dest_folder=Path(data_path).parent, progress_bar=False)
        with zipfile.ZipFile(groups_path) as groups_archive:
            group_rows = list(csv.DictReader(io.TextIOWrapper(groups_archive.open(GROUPS_FILE), encoding="utf-8")))
        group_of_image = {row["filename"]: int(row["group_id"]) for row in group_rows}
        group_split = split_groups(group_rows)

        with zipfile.ZipFile(data_path) as archive:
            for file_name in sorted(archive.namelist()):
                if not file_name.endswith(".jpeg"):
                    continue

                label_name, image_name = file_name.split("/")[-2:]
                group_id = group_of_image[image_name]
                if group_split[group_id] != split:
                    continue

                yield file_name, {"image": archive.read(file_name), "label": label_name, "group_id": group_id}

    @staticmethod
    def _labels():
        return ["colon_aca", "colon_n", "lung_aca", "lung_n", "lung_scc"]
