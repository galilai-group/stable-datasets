import io
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image as PILImage

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
from stable_datasets.splits import Split
from stable_datasets.utils import BaseDatasetBuilder, bulk_download


class PlantVillage(BaseDatasetBuilder):
    """Load PlantVillage color images using the authors' official train/test manifests."""

    VERSION = Version("1.0.0")
    BUILDER_CONFIGS = [BuilderConfig(name="color")]
    DEFAULT_CONFIG_NAME = "color"

    SOURCE = DatasetSource(
        homepage="https://github.com/spMohanty/PlantVillage-Dataset",
        citation="""@article{Mohanty2016PlantVillage,
        title={Using Deep Learning for Image-Based Plant Disease Detection},
        author={Mohanty, Sharada P. and Hughes, David P. and Salathe, Marcel},
        journal={Frontiers in Plant Science},
        volume={7},
        year={2016},
        doi={10.3389/fpls.2016.01419}
        }""",
        assets={
            split: DownloadInfo(
                url=f"https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/splits/color_{split}.txt",
                filename=f"plant-village-color-{split}.txt",
            )
            for split in (Split.TRAIN, Split.TEST)
        },
    )

    IMAGE_ARCHIVE = DownloadInfo(url="https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip")

    @staticmethod
    def _labels():
        return [
            "Apple___Apple_scab",
            "Apple___Black_rot",
            "Apple___Cedar_apple_rust",
            "Apple___healthy",
            "Blueberry___healthy",
            "Cherry_(including_sour)___Powdery_mildew",
            "Cherry_(including_sour)___healthy",
            "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
            "Corn_(maize)___Common_rust_",
            "Corn_(maize)___Northern_Leaf_Blight",
            "Corn_(maize)___healthy",
            "Grape___Black_rot",
            "Grape___Esca_(Black_Measles)",
            "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
            "Grape___healthy",
            "Orange___Haunglongbing_(Citrus_greening)",
            "Peach___Bacterial_spot",
            "Peach___healthy",
            "Pepper,_bell___Bacterial_spot",
            "Pepper,_bell___healthy",
            "Potato___Early_blight",
            "Potato___Late_blight",
            "Potato___healthy",
            "Raspberry___healthy",
            "Soybean___healthy",
            "Squash___Powdery_mildew",
            "Strawberry___Leaf_scorch",
            "Strawberry___healthy",
            "Tomato___Bacterial_spot",
            "Tomato___Early_blight",
            "Tomato___Late_blight",
            "Tomato___Leaf_Mold",
            "Tomato___Septoria_leaf_spot",
            "Tomato___Spider_mites Two-spotted_spider_mite",
            "Tomato___Target_Spot",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
            "Tomato___Tomato_mosaic_virus",
            "Tomato___healthy",
        ]

    def _info(self):
        return DatasetInfo(
            description=(
                "PlantVillage original RGB leaf images in 38 "
                "plant-health categories, using the authors' published "
                "leaf-group-aware 80/20 train/test split."
            ),
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=self._labels()),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    @classmethod
    def _image_members(cls, archive):
        groups = defaultdict(list)
        valid_labels = set(cls._labels())

        for name in archive.namelist():
            path = PurePosixPath(name)
            parts = path.parts

            if len(parts) < 4:
                continue

            # Accept either raw/color/... or a repository prefix.
            matches = [index for index in range(len(parts) - 2) if parts[index : index + 2] == ("raw", "color")]
            if not matches:
                continue

            index = matches[0]
            if len(parts) != index + 4:
                continue

            class_name = parts[index + 2]
            if class_name not in valid_labels:
                continue

            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            if name.endswith("/"):
                continue

            groups[class_name].append(name)

        return groups

    @classmethod
    def _split_members(cls, archive, split_path):
        """Resolve an official color-image manifest against the source archive."""
        members = {
            PurePosixPath(*PurePosixPath(name).parts[-4:]).as_posix(): name
            for images in cls._image_members(archive).values()
            for name in images
        }
        paths = [line.strip() for line in Path(split_path).read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(paths) != len(set(paths)):
            raise ValueError("Duplicate image paths in PlantVillage split manifest.")
        missing = set(paths) - members.keys()
        if missing:
            raise ValueError(f"PlantVillage split manifest references missing color image: {min(missing)!r}")
        return sorted(members[path] for path in paths)

    def _generate_examples(self, data_path, split):
        if split not in self._candidate_splits():
            raise ValueError(f"Unknown PlantVillage split: {split}")

        # The base builder downloads the official split manifests. Both splits
        # read the same image archive, which the downloader caches after one fetch.
        archive_path = bulk_download([self.IMAGE_ARCHIVE], dest_folder=self._raw_download_dir)[0]
        with zipfile.ZipFile(archive_path) as archive:
            selected = self._split_members(archive, data_path)
            if not selected:
                raise ValueError(f"No PlantVillage color images found for split {split!r}. Check the split manifest.")

            label_to_id = {name: index for index, name in enumerate(self._labels())}

            for name in selected:
                class_name = PurePosixPath(name).parent.name
                image_bytes = archive.read(name)
                with PILImage.open(io.BytesIO(image_bytes)) as image:
                    if image.mode != "RGB":
                        buffer = io.BytesIO()
                        image.convert("RGB").save(buffer, format="PNG")
                        image_bytes = buffer.getvalue()
                yield (
                    name,
                    {
                        "image": image_bytes,
                        "label": label_to_id[class_name],
                    },
                )
