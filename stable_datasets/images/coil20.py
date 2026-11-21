import io
import re
import zipfile
from pathlib import PurePosixPath

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


class COIL20(BaseDatasetBuilder):
    """COIL-20 images of 20 objects photographed from multiple views."""

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php",
        license=(
            "Unspecified: no dataset-specific license identified in the official source. "
            "See the documentation for source links and usage information."
        ),
        citation="""@techreport{nene1996coil20,
            title={Columbia Object Image Library (COIL-20)},
            author={Nene, Sameer A. and Nayar, Shree K. and Murase, Hiroshi},
            institution={Columbia University},
            year={1996}
        }""",
        assets={
            "train": DownloadInfo(
                url="https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip"
            ),
        },
    )

    def _info(self):
        return DatasetInfo(
            description="Processed COIL-20 object images and their view identifiers.",
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=[f"obj{i}" for i in range(1, 21)]),
                    "view_id": Value("int32"),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
            license=self.SOURCE["license"],
        )

    def _generate_examples(self, data_path, split):
        with zipfile.ZipFile(data_path) as archive:
            for member in sorted(archive.namelist()):
                filename = PurePosixPath(member).name
                match = re.fullmatch(r"obj(\d+)__(\d+)\.png", filename)
                if match is None:
                    continue

                object_id = int(match.group(1))
                view_id = int(match.group(2))
                if not 1 <= object_id <= 20:
                    continue

                with archive.open(member) as image_file:
                    image_bytes = image_file.read()

                image = PILImage.open(io.BytesIO(image_bytes)).convert("L")

                yield (
                    member,
                    {
                        "image": image,
                        "label": f"obj{object_id}",
                        "view_id": view_id,
                    },
                )
