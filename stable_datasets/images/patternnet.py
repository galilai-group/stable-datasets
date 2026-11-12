import re
import zipfile

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


# The authors' release, linked from the homepage.
_URL = (
    "https://nuisteducn1-my.sharepoint.com/:u:/g/personal/zhouwx_nuist_edu_cn/"
    "EYSPYqBztbBBqS27B7uM_mEB3R9maNJze8M1Qg9Q6cnPBQ?e=MSf977&download=1"
)
_DOWNLOAD = DownloadInfo(
    url=_URL,
    checksum="sha256:456dd031950b0429518b8ec7d30a5c4b3f6456261ee1dd573a3eee43efb29958",
    filename="PatternNet.zip",
)


CLASS_NAMES = [
    "airplane",
    "baseball field",
    "basketball court",
    "beach",
    "bridge",
    "cemetery",
    "chaparral",
    "christmas tree farm",
    "closed road",
    "coastal mansion",
    "crosswalk",
    "dense residential",
    "ferry terminal",
    "football field",
    "forest",
    "freeway",
    "golf course",
    "harbor",
    "intersection",
    "mobile home park",
    "nursing home",
    "oil gas field",
    "oil well",
    "overpass",
    "parking lot",
    "parking space",
    "railway",
    "river",
    "runway",
    "runway marking",
    "shipping yard",
    "solar panel",
    "sparse residential",
    "storage tank",
    "swimming pool",
    "tennis court",
    "transformer station",
    "wastewater treatment plant",
]

# Each class has images numbered 001-800; this builder puts 001-640 in train and 641-800 in test.
TRAIN_IMAGES_PER_CLASS = 640

_IMAGE_RE = re.compile(r"^PatternNet/images/(?P<class_dir>[a-z_]+)/[a-z_]+(?P<number>\d+)\.jpg$")


class PatternNet(BaseDatasetBuilder):
    """
    PatternNet Remote Sensing Image Classification Dataset.

    PatternNet is a large-scale high-resolution remote sensing dataset collected for
    performance evaluation of remote sensing image retrieval and scene classification.
    Images are collected from Google Earth or the Google Maps API.

    The dataset contains 38 classes with 800 images per class (30,400 images total).
    Each image is 256x256 pixels in RGB format.

    PatternNet has no official split. This builder splits each class by the
    number in its file names (e.g. ``airplane001.jpg``):

    - train: images 001-640 of each class, 24,320 images
    - test: images 641-800 of each class, 6,080 images
    """

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://sites.google.com/view/zhouwx/dataset",
        assets={"train": _DOWNLOAD, "test": _DOWNLOAD},
        citation="""@article{zhou2018patternnet,
            title={PatternNet: A benchmark dataset for performance evaluation of remote sensing image retrieval},
            author={Zhou, Weixun and Newsam, Shawn and Li, Congmin and Shao, Zhenfeng},
            journal={ISPRS Journal of Photogrammetry and Remote Sensing},
            volume={145},
            pages={197--209},
            year={2018},
            publisher={Elsevier}
        }""",
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "PatternNet is a large-scale high-resolution remote sensing image dataset "
                "consisting of 30,400 RGB images across 38 classes (800 images per class, "
                "256x256 resolution), split 80/20 per class by image number into train and test."
            ),
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=CLASS_NAMES),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        """Yield the source JPEG bytes unchanged, keeping images 001-640 of each class for train and the rest for test."""
        with zipfile.ZipFile(data_path) as archive:
            for name in sorted(archive.namelist()):
                match = _IMAGE_RE.match(name)
                if match is None or (int(match["number"]) <= TRAIN_IMAGES_PER_CLASS) != (split == "train"):
                    continue
                label = CLASS_NAMES.index(match["class_dir"].replace("_", " "))
                yield name, {"image": archive.read(name), "label": label}
