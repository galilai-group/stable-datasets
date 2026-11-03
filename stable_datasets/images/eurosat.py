import zipfile

from PIL import Image as PILImage

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


class EuroSAT(BaseDatasetBuilder):
    """
    EuroSAT Dataset

    Abstract
    A land use / land cover classification dataset built from Sentinel-2
    satellite imagery, covering 10 classes with 27,000 labeled 64x64 RGB
    images in total.

    There is no official train/val/test split shipped with the archive
    (images are organized one folder per class), so this builder exposes a
    single "train" split with every image (the same pattern as AWA2).
    """

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://github.com/phelber/EuroSAT",
        citation=r"""@article{helber2019eurosat,
                         title={EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification},
                         author={Helber, Patrick and Bischke, Benjamin and Dengel, Andreas and Borth, Damian},
                         journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
                         year={2019}}""",
        assets={
            # RGB-only version (there is also a 13-band multispectral release).
            # The original host (madm.dfki.de) was unreachable (connection
            # timeout) as of 2026-10-12, so this mirror is primary with the
            # original kept as a fallback.
            "train": DownloadInfo(
                url="https://huggingface.co/datasets/torchgeo/eurosat/resolve/main/EuroSAT.zip",
                fallbacks=["https://madm.dfki.de/files/sentinel/EuroSAT.zip"],
            ),
        },
        # Imagery also falls under the Copernicus Sentinel Data Terms and Conditions.
        license="MIT License",
    )

    def _info(self):
        return DatasetInfo(
            description="EuroSAT dataset with 10 land use / land cover classes.",
            features=Features(
                {
                    "image": ImageFeature(encode_format="JPEG"),
                    "label": ClassLabel(names=self._labels()),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license=self.SOURCE["license"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        class_names = self.info.features["label"].names
        label_mapping = {name: idx for idx, name in enumerate(class_names)}

        with zipfile.ZipFile(data_path, "r") as z:
            for path in z.namelist():
                if not path.lower().endswith((".jpg", ".jpeg")):
                    continue
                # Images live at "<archive_root>/<ClassName>/<ClassName>_<idx>.jpg".
                # We key off the immediate parent folder name rather than a
                # hardcoded root (e.g. "2750/") so a differently-named release
                # of the archive still resolves correctly.
                parts = path.split("/")
                if len(parts) < 2:
                    continue
                class_name = parts[-2]
                if class_name not in label_mapping:
                    continue

                with z.open(path) as image_file:
                    image = PILImage.open(image_file).convert("RGB")
                    yield path, {"image": image, "label": label_mapping[class_name]}

    @staticmethod
    def _labels():
        """10 land use / land cover class names, matching the dataset's folder names."""
        return [
            "AnnualCrop",
            "Forest",
            "HerbaceousVegetation",
            "Highway",
            "Industrial",
            "Pasture",
            "PermanentCrop",
            "Residential",
            "River",
            "SeaLake",
        ]
