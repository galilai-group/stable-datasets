import subprocess
import zipfile
from pathlib import Path

from stable_datasets.schema import DatasetInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, KAGGLE_CLI_SETUP_INSTRUCTIONS, kaggle_cli_failure_message


class InriaAerialImageLabeling(BaseDatasetBuilder):
    """Inria Aerial Image Labeling benchmark: building vs. non-building segmentation."""

    VERSION = Version("1.0.0")
    DATASET_ID = "sagar100rathod/inria-aerial-image-labeling-dataset"

    SOURCE = {
        "homepage": "https://project.inria.fr/aerialimagelabeling/",
        # One Kaggle archive contains both official train (with GT) and test (images only).
        # Same URI for both keys; `_split_generators` still performs a single download + extract.
        "assets": {
            "train": "kaggle://sagar100rathod/inria-aerial-image-labeling-dataset",
            "test": "kaggle://sagar100rathod/inria-aerial-image-labeling-dataset",
        },
        "citation": """@inproceedings{maggiori2017dataset,
  title={Can Semantic Labeling Methods Generalize to Any City? The Inria Aerial Image Labeling Benchmark},
  author={Maggiori, Emmanuel and Tarabalka, Yuliya and Charpiat, Guillaume and Alliez, Pierre},
  booktitle={IEEE International Geoscience and Remote Sensing Symposium (IGARSS)},
  year={2017},
  organization={IEEE}
}""",
    }

    def _info(self):
        source = self._source()
        return DatasetInfo(
            description=(
                "Pixel-wise building segmentation on aerial orthophotos (0.3 m). "
                "Training tiles include public ground truth; the official test set has images only. "
                "Raw data is obtained via the Kaggle mirror; cite the original benchmark and respect its license."
            ),
            features=Features(
                {
                    "image": ImageFeature(),
                    "mask": ImageFeature(),
                }
            ),
            supervised_keys=("image", "mask"),
            homepage=source["homepage"],
            citation=source["citation"],
        )

    def _split_generators(self):
        extract_root = self._ensure_local_dataset()
        train_img, train_gt = self._discover_train_pair(extract_root)
        test_img_dir = self._discover_test_images(extract_root)
        generators = [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"train_img_dir": train_img, "train_gt_dir": train_gt},
            ),
        ]
        if test_img_dir is not None:
            generators.append(
                SplitGenerator(
                    name=Split.TEST,
                    gen_kwargs={"test_img_dir": test_img_dir},
                )
            )
        return generators

    @classmethod
    def _kaggle_slug(cls) -> str:
        """Dataset slug (last path segment); Kaggle names the zip ``{slug}.zip``."""
        return cls.DATASET_ID.split("/")[-1]

    @classmethod
    def _local_zip_path(cls, download_dir: Path) -> Path:
        return download_dir / f"{cls._kaggle_slug()}.zip"

    @classmethod
    def _local_extract_root(cls, download_dir: Path) -> Path:
        # Underscores only: stable folder name next to the downloaded zip.
        return download_dir / f"{cls._kaggle_slug().replace('-', '_')}_extract"

    def _ensure_local_dataset(self) -> Path:
        download_dir = getattr(self, "_raw_download_dir", None)
        if download_dir is None:
            raise RuntimeError("Raw download directory is not initialized.")
        download_dir = Path(download_dir)
        extract_root = self._local_extract_root(download_dir)

        if extract_root.exists() and any(extract_root.iterdir()):
            return extract_root

        zip_path = self._local_zip_path(download_dir)
        if not zip_path.exists():
            self._download_from_kaggle(download_dir)

        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(extract_root)
        return extract_root

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

    @staticmethod
    def _list_tiffs(dir_path: Path) -> list[Path]:
        if not dir_path.is_dir():
            return []
        return sorted(p for p in dir_path.iterdir() if p.suffix.lower() in {".tif", ".tiff"})

    def _discover_train_pair(self, root: Path) -> tuple[Path, Path]:
        """Find (images_dir, gt_dir) for the training subset."""
        gt_names = {"gt", "gts", "masks", "labels", "label"}
        for gt_dir in sorted(root.rglob("*")):
            if not gt_dir.is_dir() or gt_dir.name.lower() not in gt_names:
                continue
            path_lower = str(gt_dir).lower().replace("\\", "/")
            if "train" not in path_lower:
                continue
            parent = gt_dir.parent
            for name in ("images", "img", "image", "Images", "IMG", "ortho"):
                img_dir = parent / name
                if img_dir.is_dir() and self._list_tiffs(img_dir) and self._list_tiffs(gt_dir):
                    return img_dir, gt_dir
        raise RuntimeError(
            f"Could not find training image/gt directories under {root}. "
            "Expected a layout like .../train/images/ and .../train/gt/ with matching .tif names."
        )

    def _discover_test_images(self, root: Path) -> Path | None:
        """Find test ortho directory (no public GT in the official benchmark)."""
        candidates: list[Path] = []
        for img_dir in sorted(root.rglob("*")):
            if not img_dir.is_dir():
                continue
            path_lower = str(img_dir).lower().replace("\\", "/")
            if "test" not in path_lower:
                continue
            if img_dir.name.lower() not in ("images", "img", "image", "ortho"):
                continue
            tifs = self._list_tiffs(img_dir)
            if tifs:
                candidates.append(img_dir)
        if not candidates:
            return None
        return max(candidates, key=lambda p: len(self._list_tiffs(p)))

    def _generate_examples(self, train_img_dir=None, train_gt_dir=None, test_img_dir=None, **_kwargs):
        if train_img_dir is not None and train_gt_dir is not None:
            images = {p.stem: p for p in self._list_tiffs(Path(train_img_dir))}
            masks = {p.stem: p for p in self._list_tiffs(Path(train_gt_dir))}
            for stem in sorted(set(images) & set(masks)):
                yield f"train/{stem}", {"image": str(images[stem]), "mask": str(masks[stem])}
            return

        if test_img_dir is not None:
            for path in self._list_tiffs(Path(test_img_dir)):
                yield f"test/{path.stem}", {"image": str(path), "mask": None}
            return

        raise ValueError("InriaAerialImageLabeling._generate_examples: missing split kwargs.")
