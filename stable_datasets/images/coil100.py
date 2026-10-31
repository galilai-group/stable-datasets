"""Columbia Object Image Library (COIL-100)."""

import re
import zipfile
from pathlib import PurePosixPath

from PIL import Image as PILImage

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Value, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


_IMAGE_NAME = re.compile(r"obj(\d+)__(\d+)\.png")


class COIL100(BaseDatasetBuilder):
    """100 physical objects, each photographed at 72 viewing angles."""

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php",
        assets={
            "train": DownloadInfo(
                url="https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip"
            )
        },
        citation="""@techreport{nene1996coil100,
  author = {Nene, Sameer A. and Nayar, Shree K. and Murase, Hiroshi},
  title = {Columbia Object Image Library (COIL-100)},
  institution = {Columbia University},
  number = {CUCS-006-96},
  month = feb,
  year = {1996}
}""",
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "COIL-100 contains 7,200 RGB images of 100 physical objects at 128x128 resolution: "
                "72 views per object in 5-degree increments. The single train split contains the "
                "complete official dataset; Columbia provides no predefined classification split."
            ),
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=[f"obj{object_id}" for object_id in range(1, 101)]),
                    "angle": Value("int32"),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        with zipfile.ZipFile(data_path) as archive:
            views = []
            for member in archive.infolist():
                if member.is_dir():
                    continue
                match = _IMAGE_NAME.fullmatch(PurePosixPath(member.filename).name)
                if match:
                    views.append((int(match[1]), int(match[2]), member))

            # Sort by object and angle so ZIP entry order cannot change row order.
            for index, (object_id, angle, member) in enumerate(sorted(views, key=lambda view: view[:2])):
                with archive.open(member) as image_file, PILImage.open(image_file) as source_image:
                    image = source_image.convert("RGB")
                # Filenames use object IDs 1-100; labels use indices 0-99.
                yield index, {"image": image, "label": object_id - 1, "angle": angle}
