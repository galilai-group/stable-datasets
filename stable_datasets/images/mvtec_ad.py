"""MVTec-AD anomaly detection dataset."""

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
from stable_datasets.utils import BaseDatasetBuilder

from ._anomaly_utils import ANOMALY_LABEL_NAMES, ensure_extracted, iter_images


MVTEC_AD_VERSION = Version("1.0.0")

# Textures first, then objects, matching the grouping used in the MVTec-AD paper.
MVTEC_AD_CATEGORIES = [
    "carpet",
    "grid",
    "leather",
    "tile",
    "wood",
    "bottle",
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
]

# MVTec publishes one archive per category alongside the full bundle, so loading a
# single category downloads ~100-800 MB instead of the ~5 GB bundle.
_CATEGORY_URLS = {
    "bottle": "https://www.mydrive.ch/shares/150452/132a93367fb17cdf968dfb5c4013f6e7/download/420937370-1629958698/bottle.tar.xz",
    "cable": "https://www.mydrive.ch/shares/150453/10f960e07fec2838b2cd512586633a32/download/420937413-1629958794/cable.tar.xz",
    "capsule": "https://www.mydrive.ch/shares/150454/e0ce6dd74eb150f46c0d98131b3703f2/download/420937454-1629958872/capsule.tar.xz",
    "carpet": "https://www.mydrive.ch/shares/150455/eac7fbce84d93a5094e13f391170eca4/download/420937484-1629959013/carpet.tar.xz",
    "grid": "https://www.mydrive.ch/shares/150456/bb0b2e3dc804ccb8b4485e01f8e4493b/download/420937487-1629959044/grid.tar.xz",
    "hazelnut": "https://www.mydrive.ch/shares/150457/51d49f65e84bdc100ff8035d7dd783ea/download/420937545-1629959162/hazelnut.tar.xz",
    "leather": "https://www.mydrive.ch/shares/150458/923030ce14e10a7d147e95d0f8885f6b/download/420937607-1629959262/leather.tar.xz",
    "metal_nut": "https://www.mydrive.ch/shares/150459/c68856a21dca589b0f8ff6d4ee0f18f4/download/420937637-1629959294/metal_nut.tar.xz",
    "pill": "https://www.mydrive.ch/shares/150460/d4f1c04da67034ccc8d8b5fa3e73f244/download/420938129-1629960351/pill.tar.xz",
    "screw": "https://www.mydrive.ch/shares/150461/242f454cc6385e5693c4fd4b94567d1e/download/420938130-1629960389/screw.tar.xz",
    "tile": "https://www.mydrive.ch/shares/150462/5479f0fdc97bc6fa16eab0cb0cf0109f/download/420938133-1629960456/tile.tar.xz",
    "toothbrush": "https://www.mydrive.ch/shares/150463/895a1a5fb84b958f07417784b060434b/download/420938134-1629960477/toothbrush.tar.xz",
    "transistor": "https://www.mydrive.ch/shares/150464/99b8ea0332438d64fcc2475ee73f9c29/download/420938166-1629960554/transistor.tar.xz",
    "wood": "https://www.mydrive.ch/shares/150465/d5b4115b720cdb54d217e75636e6e374/download/420938383-1629960649/wood.tar.xz",
    "zipper": "https://www.mydrive.ch/shares/150466/bd155b557520edaf692d9bfdb915c24a/download/420938385-1629960680/zipper.tar.xz",
}

_BUNDLE_URL = "https://www.mydrive.ch/shares/150996/b52ecdcbf521176e9db9c731f2304b27/download/420938113-1629960298/mvtec_anomaly_detection.tar.xz"


class MVTecADConfig(BuilderConfig):
    """BuilderConfig carrying the category a variant restricts to (None means all)."""

    def __init__(self, *, category: str | None, **kwargs):
        super().__init__(version=MVTEC_AD_VERSION, **kwargs)
        self.category = category


class MVTecAD(BaseDatasetBuilder):
    """MVTec-AD: industrial anomaly detection over 15 object and texture categories.

    The train split holds only defect-free images; the test split mixes defect-free
    images with 73 defect types, each accompanied by a pixel-level ground-truth mask.
    Select one category with ``config_name=`` (downloads that category's archive only)
    or ``config_name="all"`` for the full bundle.
    """

    VERSION = MVTEC_AD_VERSION

    BUILDER_CONFIGS = [
        MVTecADConfig(name="all", description="All 15 MVTec-AD categories", category=None),
        *[
            MVTecADConfig(name=category, description=f"MVTec-AD {category} category", category=category)
            for category in MVTEC_AD_CATEGORIES
        ],
    ]
    DEFAULT_CONFIG_NAME = "all"

    def _source(self) -> DatasetSource:
        category = self.config.category
        url = _BUNDLE_URL if category is None else _CATEGORY_URLS[category]
        # One archive holds both splits; identical URLs are deduplicated by
        # _split_generators so the archive is fetched once.
        return DatasetSource(
            homepage="https://www.mvtec.com/company/research/datasets/mvtec-ad",
            assets={
                "train": DownloadInfo(url=url),
                "test": DownloadInfo(url=url),
            },
            license=(
                "CC BY-NC-SA 4.0. Free for non-commercial research and educational use; "
                "see https://www.mvtec.com/company/research/datasets/mvtec-ad for the full terms."
            ),
            citation="""@inproceedings{bergmann2019mvtec,
  title={MVTec AD--A comprehensive real-world dataset for unsupervised anomaly detection},
  author={Bergmann, Paul and Fauser, Michael and Sattlegger, David and Steger, Carsten},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9592--9600},
  year={2019}
}""",
        )

    def _info(self):
        source = self._source()
        return DatasetInfo(
            description=(
                "MVTec-AD is an industrial anomaly detection benchmark of 5,354 high-resolution "
                "images across 15 object and texture categories. Training images are defect-free; "
                "test images cover 73 defect types with pixel-precise ground-truth masks."
            ),
            features=Features(
                {
                    "image": Image(),
                    # None for defect-free images: only anomalous test images ship a mask.
                    "mask": Image(),
                    "label": ClassLabel(names=ANOMALY_LABEL_NAMES),
                    "defect_type": Value("string"),
                    "category": ClassLabel(names=MVTEC_AD_CATEGORIES),
                    "image_path": Value("string"),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=source["homepage"],
            license=source["license"],
            citation=source["citation"],
        )

    def _categories(self) -> list[str]:
        return MVTEC_AD_CATEGORIES if self.config.category is None else [self.config.category]

    def _generate_examples(self, data_path, split):
        extract_dir = Path(self._raw_download_dir) / "mvtec_ad" / self.config.name
        root = ensure_extracted(Path(data_path), extract_dir)

        category_index = {name: idx for idx, name in enumerate(MVTEC_AD_CATEGORIES)}

        for category in self._categories():
            category_dir = root / category
            if not category_dir.is_dir():
                raise FileNotFoundError(f"Expected MVTec-AD category directory at {category_dir}.")

            split_dir = category_dir / ("train" if split == "train" else "test")
            ground_truth_dir = category_dir / "ground_truth"

            # Sorted so cache row order is reproducible across machines.
            for defect_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                defect_type = defect_dir.name
                is_anomalous = defect_type != "good"

                for image_path in iter_images(defect_dir):
                    mask_path = ground_truth_dir / defect_type / f"{image_path.stem}_mask.png"
                    yield (
                        f"{category}/{split}/{defect_type}/{image_path.name}",
                        {
                            "image": image_path,
                            "mask": mask_path if is_anomalous and mask_path.is_file() else None,
                            "label": int(is_anomalous),
                            "defect_type": defect_type,
                            "category": category_index[category],
                            "image_path": str(image_path.relative_to(root)),
                        },
                    )
