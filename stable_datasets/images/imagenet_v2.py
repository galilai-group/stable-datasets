"""ImageNetV2: three new ImageNet-1K test sets (Recht et al., ICML 2019)."""

import tarfile
from pathlib import PurePosixPath

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

from ._imagenet_rgb import to_rgb_bytes
from ._imagenet_wnids import IN1K_CLASSES


IMAGENET_V2_VERSION = Version("1.0.0")

_BASE_URL = "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main"

_CITATION = """@inproceedings{recht2019imagenet,
  title={Do ImageNet Classifiers Generalize to ImageNet?},
  author={Recht, Benjamin and Roelofs, Rebecca and Schmidt, Ludwig and Shankar, Vaishaal},
  booktitle={International Conference on Machine Learning},
  pages={5389--5400},
  year={2019}
}"""


class ImageNetV2Config(BuilderConfig):
    """BuilderConfig naming the archive of one ImageNetV2 variant and its SHA-256."""

    def __init__(self, *, archive: str, checksum: str, **kwargs):
        super().__init__(version=IMAGENET_V2_VERSION, **kwargs)
        self.archive = archive
        self.checksum = checksum


class ImageNetV2(BaseDatasetBuilder):
    """Test-only ImageNet-1K test sets collected anew, labeled in the ImageNet-1K class order."""

    VERSION = IMAGENET_V2_VERSION

    BUILDER_CONFIGS = [
        ImageNetV2Config(
            name="matched-frequency",
            description="ImageNetV2 MatchedFrequency (imagenetv2-b-33)",
            archive="imagenetv2-matched-frequency.tar.gz",
            checksum="sha256:f0c37fdf925916b19ea1323cd9a2208cdb6959ba2c32eef2a7fc393835c9ca7c",
        ),
        ImageNetV2Config(
            name="threshold-0.7",
            description="ImageNetV2 Threshold0.7 (imagenetv2-a-44)",
            archive="imagenetv2-threshold0.7.tar.gz",
            checksum="sha256:32142bfe998f9cfa4141be691f6a80eeb1e27b24bd532a668eb08c26d98897a2",
        ),
        ImageNetV2Config(
            name="top-images",
            description="ImageNetV2 TopImages (imagenetv2-c-12)",
            archive="imagenetv2-top-images.tar.gz",
            checksum="sha256:85e4ace30f3f0d660152062254b8759dda66267b27f7ec0fd8d5768f1073a04b",
        ),
    ]
    DEFAULT_CONFIG_NAME = "matched-frequency"

    def _source(self) -> DatasetSource:
        return DatasetSource(
            homepage="https://github.com/modestyachts/ImageNetV2",
            assets={
                "test": DownloadInfo(url=f"{_BASE_URL}/{self.config.archive}", checksum=self.config.checksum),
            },
            citation=_CITATION,
            license="MIT (code); images from Flickr, usable under the same terms as ImageNet",
        )

    def _info(self):
        source = self._source()
        return DatasetInfo(
            description=f"{self.config.description}: 10,000 test images, 10 for each ImageNet-1K class.",
            features=Features({"image": Image(encode_format="JPEG"), "label": ClassLabel(names=IN1K_CLASSES)}),
            supervised_keys=("image", "label"),
            homepage=source["homepage"],
            citation=source["citation"],
            license=source["license"],
        )

    def _generate_examples(self, data_path, split):
        # Archive layout: <top dir>/<class index>/<hash>.jpeg, with class folders named by the
        # ImageNet-1K class index ("0" to "999"), not by WordNet ID. Despite the .tar.gz name the
        # files are uncompressed tars, so "r|*" lets tarfile detect the compression.
        with tarfile.open(data_path, mode="r|*") as archive:
            for member in archive:
                if not member.isfile():
                    continue

                folder = PurePosixPath(member.name).parent.name
                if not (folder.isascii() and folder.isdigit()) or int(folder) >= len(IN1K_CLASSES):
                    raise ValueError(f"Expected a class folder named 0-999 in the archive, got {member.name!r}.")

                image_bytes = archive.extractfile(member).read()
                yield member.name, {"image": to_rgb_bytes(image_bytes), "label": int(folder)}
