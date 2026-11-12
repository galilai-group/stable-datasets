"""VisA (Visual Anomaly) dataset."""

import csv
from pathlib import Path

from stable_datasets.schema import (
    BuilderConfig,
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
from stable_datasets.utils import BaseDatasetBuilder, _default_dest_folder, bulk_download

from ._anomaly_utils import ANOMALY_LABEL_NAMES, ensure_extracted


VISA_VERSION = Version("1.0.0")

VISA_OBJECTS = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]

_IMAGES_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"

# The image archive ships no split definition. The official 1-class split lives in the
# spot-diff repository and is what the VisA paper evaluates on: 8,659 train rows (all
# normal) and 2,162 test rows (962 normal + 1,200 anomalous, each with a mask).
_SPLIT_CSV_URL = "https://raw.githubusercontent.com/amazon-science/spot-diff/main/split_csv/1cls.csv"


class VisAConfig(BuilderConfig):
    """BuilderConfig carrying the object a variant restricts to (None means all)."""

    def __init__(self, *, object_name: str | None, **kwargs):
        super().__init__(version=VISA_VERSION, **kwargs)
        self.object_name = object_name


class VisA(BaseDatasetBuilder):
    """VisA: 10,821 images over 12 objects for visual anomaly detection.

    Follows the official 1-class protocol: the train split contains only normal
    images, and the test split mixes normal images with anomalous ones carrying
    pixel-level masks. Select one object with ``config_name=`` or use
    ``config_name="all"`` for all twelve.

    The images ship as a single 1.9 GB archive covering every object, so a
    per-object config still downloads the full archive but caches only that
    object's rows.
    """

    VERSION = VISA_VERSION

    BUILDER_CONFIGS = [
        VisAConfig(name="all", description="All 12 VisA objects", object_name=None),
        *[
            VisAConfig(name=object_name, description=f"VisA {object_name} object", object_name=object_name)
            for object_name in VISA_OBJECTS
        ],
    ]
    DEFAULT_CONFIG_NAME = "all"

    SOURCE = DatasetSource(
        homepage="https://github.com/amazon-science/spot-diff",
        assets={
            "images": DownloadInfo(url=_IMAGES_URL, filename="VisA_20220922.tar"),
            "split_csv": DownloadInfo(url=_SPLIT_CSV_URL, filename="visa_1cls.csv"),
        },
        license="CC BY 4.0",
        citation="""@inproceedings{zou2022spot,
  title={SPot-the-Difference Self-supervised Pre-training for Anomaly Detection and Segmentation},
  author={Zou, Yang and Jeong, Jongheon and Pei, Latha and Zhang, Xiaolong and Cheng, Wenchao and Li, Xin},
  booktitle={European Conference on Computer Vision},
  pages={392--408},
  year={2022},
  organization={Springer}
}""",
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "VisA contains 10,821 high-resolution images across 12 objects spanning printed "
                "circuit boards, multi-instance items and single instances. Anomalous images cover "
                "surface and structural defects and carry pixel-level segmentation masks."
            ),
            features=Features(
                {
                    "image": Image(),
                    # None for normal images: only anomalous rows ship a mask.
                    "mask": Image(),
                    "label": ClassLabel(names=ANOMALY_LABEL_NAMES),
                    "object": ClassLabel(names=VISA_OBJECTS),
                    "image_path": Value("string"),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license=self.SOURCE["license"],
            citation=self.SOURCE["citation"],
        )

    def _candidate_splits(self):
        # SOURCE assets are named by role, not by split, so the default
        # asset-key-is-split-name inference does not apply here.
        return [Split.TRAIN, Split.TEST]

    def _split_generators(self):
        source = self._source()
        assets = source["assets"]
        download_dir = Path(getattr(self, "_raw_download_dir", None) or _default_dest_folder())

        images_archive, split_csv = bulk_download(
            [assets["images"], assets["split_csv"]],
            dest_folder=download_dir,
        )

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"data_path": images_archive, "split_csv": split_csv, "split": "train"},
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={"data_path": images_archive, "split_csv": split_csv, "split": "test"},
            ),
        ]

    def _generate_examples(self, data_path, split_csv, split):
        extract_dir = Path(self._raw_download_dir) / "visa"
        root = ensure_extracted(Path(data_path), extract_dir)

        object_index = {name: idx for idx, name in enumerate(VISA_OBJECTS)}
        wanted = self.config.object_name

        with open(split_csv, newline="") as handle:
            rows = list(csv.DictReader(handle))

        # Sorted so cache row order does not depend on CSV ordering upstream.
        rows = sorted(
            (row for row in rows if row["split"] == split and (wanted is None or row["object"] == wanted)),
            key=lambda row: (row["object"], row["label"], row["image"]),
        )
        if not rows:
            raise ValueError(f"No VisA rows for split={split!r}, object={wanted!r} in {split_csv}.")

        for row in rows:
            image_path = root / row["image"]
            if not image_path.is_file():
                raise FileNotFoundError(f"VisA image listed in the split CSV is missing: {image_path}")

            mask_rel = row["mask"].strip()
            mask_path = root / mask_rel if mask_rel else None

            yield (
                row["image"],
                {
                    "image": image_path,
                    "mask": mask_path if mask_path is not None and mask_path.is_file() else None,
                    "label": int(row["label"] == "anomaly"),
                    "object": object_index[row["object"]],
                    "image_path": row["image"],
                },
            )
