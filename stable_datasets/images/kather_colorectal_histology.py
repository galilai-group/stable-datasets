"""The eight-class Kather colorectal histology texture dataset."""

import io
import re
from zipfile import ZipFile

from PIL import Image as PILImage

from stable_datasets.schema import (
    ClassLabel,
    DatasetInfo,
    DatasetSource,
    DownloadInfo,
    Features,
    Value,
    Version,
)
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


class KatherColorectalHistology(BaseDatasetBuilder):
    """
    Balanced classification of 5,000 colorectal histology tissue tiles from the
    Kather dataset. The dataset contains 625 examples in each of eight classes.

    Only train is provided: the release has no official train/test split.
    TIFF tiles are stored as RGB PNGs without resizing. `source_image_id`
    preserves the source image name embedded in each tile's filename, while
    `slide_id` is its numeric `CRC-Prim-HE-NN` prefix, grouping letter-suffixed
    variants (for example, `CRC-Prim-HE-10c`) under that prefix. These groups
    can be used to keep related tiles together when constructing splits;
    they are not independently verified patient identifiers.
    """

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://zenodo.org/records/53169",
        license="CC BY 4.0",
        citation="""@article{kather2016multi,
            title={Multi-class texture analysis in colorectal cancer histology},
            author={Kather, Jakob Nikolas and Weis, Cleo-Aron and Bianconi, Francesco
                    and Melchers, Susanne M and Schad, Lothar R and Gaiser, Timo
                    and Marx, Alexander and Z{\\\"o}llner, Frank Gerrit},
            journal={Scientific Reports},
            volume={6},
            pages={27988},
            year={2016},
            doi={10.1038/srep27988}
        }""",
        assets={
            "train": DownloadInfo(
                url="https://zenodo.org/records/53169/files/Kather_texture_2016_image_tiles_5000.zip",
                checksum="md5:0ddbebfc56344752028fda72602aaade",
            ),
        },
    )

    def _info(self):
        source = self._source()
        return DatasetInfo(
            description=(
                "Kather Colorectal Histology: 5,000 H&E-stained 150x150 RGB tissue tiles, "
                "with 625 examples in each of eight classes. Only train is provided; "
                "filename-derived source-image and slide groups are included for constructing "
                "splits."
            ),
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=self._labels()),
                    "filename": Value("string"),
                    "source_image_id": Value("string"),
                    "slide_id": Value("string"),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=source.homepage,
            citation=source.citation,
            license=source.license,
        )

    def _generate_examples(self, data_path=None, **_kwargs):
        """Yield sorted tile paths, losslessly encoded images, and source groups."""
        if data_path is None:
            raise TypeError("data_path is required")
        class_directories = {
            f"{index:02d}_{label.upper()}": label
            for index, label in enumerate(self._labels(), start=1)
        }
        with ZipFile(data_path) as archive:
            for member in sorted(archive.infolist(), key=lambda entry: entry.filename):
                parts = member.filename.split("/")
                if (
                    member.is_dir()
                    or len(parts) != 3
                    or parts[0] != "Kather_texture_2016_image_tiles_5000"
                    or parts[1] not in class_directories
                    or not parts[2].endswith(".tif")
                ):
                    continue

                filename = parts[2]
                match = re.fullmatch(
                    r"[^_]+_((CRC-Prim-HE-\d{2})[a-z]?(?:_[^.]+)?)\.tif_Row_\d+_Col_\d+\.tif",
                    filename,
                )
                if match is None:
                    raise ValueError(
                        f"Cannot identify source image for {member.filename}"
                    )

                try:
                    with PILImage.open(
                        io.BytesIO(archive.read(member))
                    ) as source_image:
                        image = source_image.convert("RGB")
                except (OSError, ValueError) as error:
                    raise ValueError(
                        f"Cannot decode Kather image {member.filename}"
                    ) from error
                if image.size != (150, 150):
                    raise ValueError(
                        f"Expected a 150x150 Kather tile for {member.filename}, got {image.size}"
                    )

                # Supplying bytes avoids `ImageFeature` preserving the original TIFF encoding.
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                yield (
                    member.filename,
                    {
                        "image": buffer.getvalue(),
                        "label": class_directories[parts[1]],
                        "filename": filename,
                        "source_image_id": match.group(1),
                        "slide_id": match.group(2),
                    },
                )

    @staticmethod
    def _labels():
        return [
            "tumor",
            "stroma",
            "complex",
            "lympho",
            "debris",
            "mucosa",
            "adipose",
            "empty",
        ]
