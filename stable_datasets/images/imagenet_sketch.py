"""ImageNet-Sketch: black-and-white sketches of the 1000 ImageNet-1K classes."""

import posixpath
import zipfile

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Image, Version
from stable_datasets.utils import BaseDatasetBuilder

from ._imagenet_rgb import to_rgb_bytes
from ._imagenet_wnids import IN1K_CLASSES, WNID_TO_IDX


_ZIP_SHA256 = "sha256:4d3ce6b82a7c48aabf14aeac105eb533e75707f152cc858f55bfc022031a72ed"


class ImageNetSketch(BaseDatasetBuilder):
    """Test-only sketch images, labeled in the ImageNet-1K class order."""

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://github.com/HaohanWang/ImageNet-Sketch",
        assets={
            "test": DownloadInfo(
                url="https://huggingface.co/datasets/songweig/imagenet_sketch/resolve/main/data/ImageNet-Sketch.zip",
                checksum=_ZIP_SHA256,
            ),
        },
        citation="""@inproceedings{wang2019learning,
                      title={Learning Robust Global Representations by Penalizing Local Predictive Power},
                      author={Wang, Haohan and Ge, Songwei and Lipton, Zachary and Xing, Eric P},
                      booktitle={Advances in Neural Information Processing Systems},
                      pages={10506--10518},
                      year={2019}}""",
        license="MIT",
    )

    def _info(self):
        return DatasetInfo(
            description="""ImageNet-Sketch contains 50,889 black-and-white sketches of the 1000 ImageNet-1K
                classes, collected with Google Image queries "sketch of __" and cleaned manually. Classes
                with fewer than 50 images were augmented by flipping and rotating. It is a test-only
                benchmark for out-of-domain generalization of ImageNet-1K classifiers.""",
            features=Features({"image": Image(encode_format="JPEG"), "label": ClassLabel(names=IN1K_CLASSES)}),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
            license=self.SOURCE["license"],
        )

    def _generate_examples(self, data_path, split):
        # Archive layout: sketch/<wnid>/<name>.JPEG. The 7.6 GB zip is read in place, and entries
        # are sorted so the row order does not depend on how the archive was written.
        with zipfile.ZipFile(data_path, "r") as archive:
            for entry in sorted(archive.infolist(), key=lambda info: info.filename):
                if entry.is_dir():
                    continue
                name = entry.filename

                wnid = posixpath.basename(posixpath.dirname(name)).lower()
                if wnid not in WNID_TO_IDX:
                    raise ValueError(f"ImageNet-Sketch entry {name!r} is not inside an ImageNet-1K WNID folder.")

                # 8,850 of the 50,889 files are grayscale JPEGs.
                yield name, {"image": to_rgb_bytes(archive.read(entry)), "label": WNID_TO_IDX[wnid]}
