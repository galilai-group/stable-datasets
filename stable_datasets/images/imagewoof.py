"""Full-resolution Imagewoof with original labels and official splits."""

import tarfile
from collections.abc import Iterator
from pathlib import Path

from PIL import Image as PILImage

from stable_datasets.schema import (
    BuilderConfig,
    ClassLabel,
    DatasetInfo,
    DatasetSource,
    DownloadInfo,
    Features,
    Image,
    Version,
)
from stable_datasets.utils import BaseDatasetBuilder


# Fixed label order, verified against both split directories in imagewoof2.tgz.
_IMAGEWOOF_CLASSES = [
    "n02086240",
    "n02087394",
    "n02088364",
    "n02089973",
    "n02093754",
    "n02096294",
    "n02099601",
    "n02105641",
    "n02111889",
    "n02115641",
]


class Imagewoof(BaseDatasetBuilder):
    """Ten dog classes from ImageNet, using the official Imagewoof v2 archive."""

    VERSION = Version("1.0.0")
    BUILDER_CONFIGS = [BuilderConfig(name="default", description="Full-resolution Imagewoof v2, original labels.")]
    SOURCE = DatasetSource(
        homepage="https://github.com/fastai/imagenette#imagewoof",
        assets={
            "train": DownloadInfo(url="https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2.tgz"),
            "val": DownloadInfo(url="https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2.tgz"),
        },
        citation="""@software{Howard_Imagewoof_2019,
            title={Imagewoof: a subset of 10 classes from Imagenet that aren't so easy to classify},
            author={Jeremy Howard},
            year={2019},
            month={March},
            publisher={GitHub},
            url={https://github.com/fastai/imagenette#imagewoof}
        }""",
    )

    def _info(self) -> DatasetInfo:
        """Describe the full-resolution image and original class-label fields."""
        return DatasetInfo(
            description="Full-resolution Imagewoof v2 with 10 classes and official train/validation splits.",
            features=Features({"image": Image(), "label": ClassLabel(names=_IMAGEWOOF_CLASSES)}),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path: str | Path, split: str) -> Iterator[tuple[str, dict[str, object]]]:
        """Read image members in place, convert to RGB, and map directory labels."""
        with tarfile.open(data_path, "r:*") as archive:
            for member in archive:
                if not member.isfile() or not member.name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                parts = member.name.split("/")
                if len(parts) != 4 or parts[0] != "imagewoof2" or parts[1] != split:
                    continue
                wnid, filename = parts[2:]
                if wnid not in _IMAGEWOOF_CLASSES or filename.startswith("."):
                    continue
                file_obj = archive.extractfile(member)
                if file_obj is None:
                    continue
                with file_obj, PILImage.open(file_obj) as source:
                    image = source.convert("RGB")
                label = _IMAGEWOOF_CLASSES.index(wnid)
                yield member.name, {"image": image, "label": label}
