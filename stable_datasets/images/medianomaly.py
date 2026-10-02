"""MedIAnomaly: a benchmark for anomaly detection in medical images."""

import json
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

from ._anomaly_utils import ANOMALY_LABEL_NAMES, ensure_extracted, iter_images


MEDIANOMALY_VERSION = Version("1.0.0")

# Six subsets are redistributed pre-processed on Zenodo. ISIC2018_Task3 is not:
# it is gated behind an ISIC challenge account, so it is built from data_dir=.
_ZENODO_RECORD = "12677223"
_ZENODO_SUBSETS = {
    "BrainTumor": "md5:ab6482f646262fc4669b0135d80a0482",
    "BraTS2021": "md5:237f2aee77e099f7483808ddec49a57c",
    "LAG": "md5:0b4034ae8d5275f7723a512adb218f86",
    "RSNA": "md5:5d6b85ce4b2daad8f1760eb23c050a7d",
    "VinCXR": "md5:45cf848a4eba925062531c5f2d344ddc",
    "Camelyon16": "md5:c26c026d1c315deb52720d4c195e874f",
}

_ISIC_SUBSET = "ISIC2018_Task3"

MEDIANOMALY_SUBSETS = [*_ZENODO_SUBSETS, _ISIC_SUBSET]

# Subsets whose splits are described by a data.json listing relative image names.
_DATA_JSON_SUBSETS = ("RSNA", "VinCXR", "BrainTumor", "LAG")

_ISIC_INSTRUCTIONS = (
    "MedIAnomaly's ISIC2018_Task3 subset is not redistributed on Zenodo. Download "
    "ISIC2018_Task3_Training_Input, ISIC2018_Task3_Training_GroundTruth, "
    "ISIC2018_Task3_Test_Input and ISIC2018_Task3_Test_GroundTruth from "
    "https://challenge.isic-archive.com/data/#2018 (an ISIC account and agreement to "
    "their terms are required), arrange them as:\n"
    "    <data_dir>/ISIC2018_Task3/\n"
    "        ISIC2018_Task3_Training_Input/\n"
    "        ISIC2018_Task3_Training_GroundTruth/\n"
    "        ISIC2018_Task3_Test_Input/\n"
    "        ISIC2018_Task3_Test_GroundTruth/\n"
    "then load with MedIAnomaly(config_name='ISIC2018_Task3', data_dir=<data_dir>)."
)


def _zenodo_url(filename: str) -> str:
    return f"https://zenodo.org/api/records/{_ZENODO_RECORD}/files/{filename}/content"


class MedIAnomalyConfig(BuilderConfig):
    """BuilderConfig carrying the subset a variant restricts to (None means all)."""

    def __init__(self, *, subset: str | None, **kwargs):
        super().__init__(version=MEDIANOMALY_VERSION, **kwargs)
        self.subset = subset


class MedIAnomaly(BaseDatasetBuilder):
    """MedIAnomaly: anomaly detection across seven medical imaging datasets.

    Each subset follows the one-class protocol: the train split holds only normal
    images, the test split mixes normal and abnormal ones. Six subsets download
    from Zenodo; ``ISIC2018_Task3`` is gated and must be supplied via ``data_dir=``.

    ``config_name="all"`` covers the six redistributable subsets (~2.4 GB).
    """

    VERSION = MEDIANOMALY_VERSION

    BUILDER_CONFIGS = [
        MedIAnomalyConfig(name="all", description="All six Zenodo-hosted MedIAnomaly subsets", subset=None),
        *[
            MedIAnomalyConfig(name=subset, description=f"MedIAnomaly {subset} subset", subset=subset)
            for subset in MEDIANOMALY_SUBSETS
        ],
    ]
    DEFAULT_CONFIG_NAME = "all"

    def __init__(self, config_name: str | None = None, data_dir: str | Path | None = None, **kwargs):
        self.data_dir = Path(data_dir).expanduser() if data_dir is not None else None
        super().__init__(config_name=config_name, **kwargs)

    def _subsets(self) -> list[str]:
        return list(_ZENODO_SUBSETS) if self.config.subset is None else [self.config.subset]

    def _source(self) -> DatasetSource:
        assets = {
            subset: DownloadInfo(
                url=_zenodo_url(f"{subset}.tar.gz"),
                fallbacks=[f"https://zenodo.org/records/{_ZENODO_RECORD}/files/{subset}.tar.gz?download=1"],
                checksum=checksum,
                # The API URL ends in /content, so the archive name must be explicit.
                filename=f"{subset}.tar.gz",
            )
            for subset, checksum in _ZENODO_SUBSETS.items()
            if self.config.subset is None or subset == self.config.subset
        }
        return DatasetSource(
            homepage="https://github.com/caiyu6666/MedIAnomaly",
            assets=assets,
            license="CC BY 4.0",
            citation="""@article{cai2024medianomaly,
  title={MedIAnomaly: A comparative study of anomaly detection in medical images},
  author={Cai, Yu and Zhang, Weiwen and Chen, Hao and Cheng, Kwang-Ting},
  journal={arXiv preprint arXiv:2404.04518},
  year={2024}
}""",
        )

    def _info(self):
        return DatasetInfo(
            description=(
                "MedIAnomaly benchmarks anomaly detection across seven medical imaging datasets "
                "spanning chest radiographs (RSNA, VinCXR), brain MRI (BrainTumor, BraTS2021), "
                "retinal fundus images (LAG), histopathology (Camelyon16) and dermoscopy "
                "(ISIC2018_Task3). Train splits contain only normal images."
            ),
            features=Features(
                {
                    "image": Image(),
                    "label": ClassLabel(names=ANOMALY_LABEL_NAMES),
                    "subset": ClassLabel(names=MEDIANOMALY_SUBSETS),
                    "image_path": Value("string"),
                }
            ),
            supervised_keys=("image", "label"),
            homepage="https://github.com/caiyu6666/MedIAnomaly",
            license="CC BY 4.0",
            citation=self._source()["citation"],
        )

    def _candidate_splits(self):
        # SOURCE assets are named by subset, not by split.
        return [Split.TRAIN, Split.TEST]

    def _split_generators(self):
        if self.config.subset == _ISIC_SUBSET:
            if self.data_dir is None:
                raise ValueError(_ISIC_INSTRUCTIONS)
            roots = {_ISIC_SUBSET: self.data_dir / _ISIC_SUBSET}
            if not roots[_ISIC_SUBSET].is_dir():
                raise FileNotFoundError(f"Expected {roots[_ISIC_SUBSET]} to exist.\n\n{_ISIC_INSTRUCTIONS}")
        else:
            source = self._source()
            assets = source["assets"]
            subsets = self._subsets()
            download_dir = Path(getattr(self, "_raw_download_dir", None) or _default_dest_folder())
            archives = bulk_download([assets[subset] for subset in subsets], dest_folder=download_dir)

            extract_root = Path(self._raw_download_dir) / "medianomaly"
            roots = {}
            for subset, archive in zip(subsets, archives):
                # Each archive already contains a top-level <subset>/ directory.
                ensure_extracted(Path(archive), extract_root / subset)
                roots[subset] = extract_root / subset / subset

        return [
            SplitGenerator(name=Split.TRAIN, gen_kwargs={"roots": roots, "split": "train"}),
            SplitGenerator(name=Split.TEST, gen_kwargs={"roots": roots, "split": "test"}),
        ]

    def _generate_examples(self, roots, split):
        subset_index = {name: idx for idx, name in enumerate(MEDIANOMALY_SUBSETS)}

        for subset, root in roots.items():
            root = Path(root)
            for image_path, label in self._iter_subset(subset, root, split):
                # Relative to the subset root so the cache stays portable across machines.
                relative_path = image_path.relative_to(root)
                yield (
                    f"{subset}/{split}/{image_path.name}",
                    {
                        "image": image_path,
                        "label": label,
                        "subset": subset_index[subset],
                        "image_path": f"{subset}/{relative_path}",
                    },
                )

    def _iter_subset(self, subset: str, root: Path, split: str):
        """Yield ``(path, label)`` pairs for one subset, honouring its own layout."""
        if subset in _DATA_JSON_SUBSETS:
            yield from self._iter_data_json(root, split)
        elif subset == "Camelyon16":
            # "Ungood" is this subset's name for the abnormal class.
            yield from self._iter_folders(root, split, normal="good", abnormal="Ungood")
        elif subset == "BraTS2021":
            yield from self._iter_folders(root, split, normal="normal", abnormal="tumor")
        elif subset == _ISIC_SUBSET:
            yield from self._iter_isic(root, split)
        else:
            raise ValueError(f"Unknown MedIAnomaly subset {subset!r}.")

    @staticmethod
    def _iter_data_json(root: Path, split: str):
        data_json = root / "data.json"
        if not data_json.is_file():
            raise FileNotFoundError(f"Expected MedIAnomaly data.json at {data_json}.")
        with open(data_json) as handle:
            entries = json.load(handle)[split]

        # Key "0" is the normal class, "1" the abnormal one. Train holds only "0".
        for label_key in sorted(entries):
            for name in sorted(entries[label_key]):
                yield root / "images" / name, int(label_key)

    @staticmethod
    def _iter_folders(root: Path, split: str, *, normal: str, abnormal: str):
        if split == "train":
            # BraTS2021 stores training images directly under train/; Camelyon16 nests them in good/.
            train_dir = root / "train"
            nested = train_dir / normal
            for path in iter_images(nested if nested.is_dir() else train_dir):
                yield path, 0
            return

        for path in iter_images(root / "test" / normal):
            yield path, 0
        for path in iter_images(root / "test" / abnormal):
            yield path, 1

    @staticmethod
    def _iter_isic(root: Path, split: str):
        import pandas as pd

        stage = "Training" if split == "train" else "Test"
        ground_truth = root / f"ISIC2018_Task3_{stage}_GroundTruth" / f"ISIC2018_Task3_{stage}_GroundTruth.csv"
        if not ground_truth.is_file():
            raise FileNotFoundError(f"Expected {ground_truth}.\n\n{_ISIC_INSTRUCTIONS}")

        images_dir = root / f"ISIC2018_Task3_{stage}_Input"
        table = pd.read_csv(ground_truth)

        # NV (melanocytic nevus) is the normal class; every other diagnosis is abnormal.
        # The train split keeps normal images only, matching the one-class protocol.
        normal = table[table["NV"] == 1]["image"]
        for name in sorted(normal):
            yield images_dir / f"{name}.jpg", 0

        if split == "train":
            return

        abnormal = table[table["NV"] == 0]["image"]
        for name in sorted(abnormal):
            yield images_dir / f"{name}.jpg", 1
