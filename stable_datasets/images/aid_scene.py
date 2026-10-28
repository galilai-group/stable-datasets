import subprocess
import zipfile
from pathlib import Path

from stable_datasets.schema import ClassLabel, DatasetInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.splits import SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, KAGGLE_CLI_SETUP_INSTRUCTIONS, kaggle_cli_failure_message


class AIDScene(BaseDatasetBuilder):
    """Aerial Image Dataset (AID) for scene classification."""

    VERSION = Version("1.0.0")
    DATASET_ID = "jiayuanchengala/aid-scene-classification-datasets"
    SINGLE_SPLIT = "all"
    LABELS = [
        "Airport",
        "BareLand",
        "BaseballField",
        "Beach",
        "Bridge",
        "Center",
        "Church",
        "Commercial",
        "DenseResidential",
        "Desert",
        "Farmland",
        "Forest",
        "Industrial",
        "Meadow",
        "MediumResidential",
        "Mountain",
        "Park",
        "Parking",
        "Playground",
        "Pond",
        "Port",
        "RailwayStation",
        "Resort",
        "River",
        "School",
        "SparseResidential",
        "Square",
        "Stadium",
        "StorageTanks",
        "Viaduct",
    ]

    SOURCE = {
        "homepage": "https://www.kaggle.com/datasets/jiayuanchengala/aid-scene-classification-datasets",
        "assets": {
            "all": "kaggle://jiayuanchengala/aid-scene-classification-datasets",
        },
        "citation": """@article{xia2017aid,
            title={AID: A benchmark data set for performance evaluation of aerial scene classification},
            author={Xia, Gui-Song and Hu, Jingwen and Hu, Fan and Shi, Baoguang and Bai, Xiang and Zhong, Yanfei and Zhang, Liangpei and Lu, Xiaoqiang},
            journal={IEEE Transactions on Geoscience and Remote Sensing},
            volume={55},
            number={7},
            pages={3965--3981},
            year={2017},
            publisher={IEEE}
            }""",
    }

    @classmethod
    def _kaggle_slug(cls) -> str:
        """Dataset slug (last path segment); Kaggle names the zip ``{slug}.zip``."""
        return cls.DATASET_ID.split("/")[-1]

    @classmethod
    def _local_zip_path(cls, download_dir: Path) -> Path:
        return download_dir / f"{cls._kaggle_slug()}.zip"

    @classmethod
    def _local_extract_root(cls, download_dir: Path) -> Path:
        return download_dir / f"{cls._kaggle_slug().replace('-', '_')}_extract"

    def _info(self):
        source = self._source()
        return DatasetInfo(
            description=(
                "AID (Aerial Image Dataset) is a remote sensing scene classification dataset "
                "with 30 semantic scene categories."
            ),
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=self.LABELS),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=source["homepage"],
            citation=source["citation"],
        )

    def _split_generators(self):
        image_root = self._ensure_local_dataset()
        return [SplitGenerator(name=self.SINGLE_SPLIT, gen_kwargs={"data_path": image_root})]

    def _ensure_local_dataset(self) -> Path:
        download_dir = getattr(self, "_raw_download_dir", None)
        if download_dir is None:
            raise RuntimeError("Raw download directory is not initialized.")

        download_dir = Path(download_dir)
        extract_dir = self._local_extract_root(download_dir)
        image_root = self._find_image_root(extract_dir)
        if image_root is not None:
            return image_root

        zip_path = self._local_zip_path(download_dir)
        if not zip_path.exists():
            self._download_from_kaggle(download_dir)

        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(extract_dir)

        image_root = self._find_image_root(extract_dir)
        if image_root is None:
            raise RuntimeError(f"Could not locate class-folders under extracted data at {extract_dir}.")
        return image_root

    def _download_from_kaggle(self, download_dir: Path) -> None:
        try:
            subprocess.run(
                [
                    "kaggle",
                    "datasets",
                    "download",
                    "-d",
                    self.DATASET_ID,
                    "-p",
                    str(download_dir),
                    "--force",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as error:
            raise RuntimeError(
                "Kaggle CLI was not found (the `kaggle` command is not on PATH)." + KAGGLE_CLI_SETUP_INSTRUCTIONS
            ) from error
        except subprocess.CalledProcessError as error:
            raise RuntimeError(kaggle_cli_failure_message(self.DATASET_ID, error)) from error

    def _find_image_root(self, root: Path) -> Path | None:
        if not root.exists():
            return None

        candidates = [root] + [p for p in root.rglob("*") if p.is_dir()]
        for candidate in candidates:
            class_dirs = [d for d in candidate.iterdir() if d.is_dir()]
            if len(class_dirs) < 10:
                continue
            if all(any(f.suffix.lower() in {".jpg", ".jpeg", ".png"} for f in d.iterdir()) for d in class_dirs):
                return candidate
        return None

    @staticmethod
    def _normalize_label(name: str) -> str:
        return name.replace("_", "").replace("-", "").replace(" ", "").lower()

    def _generate_examples(self, data_path):
        root = Path(data_path)
        label_map = {self._normalize_label(label): label for label in self.LABELS}

        all_files = sorted([p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        by_class: dict[str, list[Path]] = {label: [] for label in self.LABELS}

        for file_path in all_files:
            folder_name = file_path.parent.name
            normalized = self._normalize_label(folder_name)
            canonical = label_map.get(normalized)
            if canonical is not None:
                by_class[canonical].append(file_path)

        for class_name in self.LABELS:
            class_files = sorted(by_class[class_name])
            if not class_files:
                continue
            for file_path in class_files:
                rel_key = f"{class_name}/{file_path.name}"
                label = self.info.features["label"].str2int(class_name)
                yield rel_key, {"image": str(file_path), "label": label}
