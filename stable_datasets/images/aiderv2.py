"""AIDERv2: aerial images of natural disasters (earthquake, flood, wildfire) and normal scenes."""

import io
import zipfile

from PIL import Image as PILImage

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


# Label i is entry i here (alphabetical, matching the class folder names).
_AIDERV2_CLASSES: list[str] = ["Earthquake", "Flood", "Normal", "Wildfire"]


_FOLDER_TO_LABEL: dict[str, int] = {
    "earthquake": 0,
    "earthquakes": 0,
    "flood": 1,
    "floods": 1,
    "normal": 2,
    "wildfire": 3,
    "fire": 3,
}

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


class AIDERv2(BaseDatasetBuilder):
    """AIDERv2: 16,723 aerial images in 4 classes, with official train/validation/test splits.

    An extension of AIDER (Aerial Image Dataset for Emergency Response Applications). Labels are
    integers in [0, 4): 0 = Earthquake (collapsed buildings), 1 = Flood, 2 = Normal (no disaster),
    3 = Wildfire. Splits: ``train`` (13,399), ``validation`` (1,670) and ``test`` (1,654). Images
    are returned as RGB at their original sizes.

    Note: the images were gathered from several open datasets and from frames of YouTube videos, so
    the CC BY 4.0 license is the depositors' label and the underlying sources have their own terms.
    Frames from the same video may appear in more than one split.
    """

    VERSION = Version("1.0.0")

    # Each split is its own zip on Zenodo (record 10891054, version v1).
    SOURCE = DatasetSource(
        homepage="https://zenodo.org/records/10891054",
        assets={
            "train": DownloadInfo(
                url="https://zenodo.org/records/10891054/files/Train.zip?download=1",
            ),
            "validation": DownloadInfo(
                url="https://zenodo.org/records/10891054/files/Val.zip?download=1",
            ),
            "test": DownloadInfo(
                url="https://zenodo.org/records/10891054/files/Test.zip?download=1",
            ),
        },
        citation="""@inproceedings{shianios2023benchmark,
                      title={A Benchmark and Investigation of Deep-Learning-Based Techniques for Detecting Natural Disasters in Aerial Images},
                      author={Shianios, D. and Kyrkou, C. and Kolios, P. S.},
                      booktitle={Computer Analysis of Images and Patterns (CAIP)},
                      series={Lecture Notes in Computer Science},
                      volume={14185},
                      publisher={Springer},
                      year={2023},
                      doi={10.1007/978-3-031-44240-7_24}}
                    @article{shianios2024direcnetv2,
                      title={DiRecNetV2: A Transformer-Enhanced Network for Aerial Disaster Recognition},
                      author={Shianios, D. and Kolios, P. and Kyrkou, C.},
                      journal={SN Computer Science},
                      year={2024}}""",
        license="CC BY 4.0",
    )

    def _info(self):
        return DatasetInfo(
            description="""AIDERv2 contains 16,723 aerial images in four classes: earthquake (collapsed
                buildings), flood, wildfire and normal (no disaster). It extends the AIDER dataset with images
                from other open datasets and from YouTube video frames, and ships with official
                train, validation and test splits.""",
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=_AIDERV2_CLASSES),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
            license=self.SOURCE["license"],
        )

    def _generate_examples(self, data_path, split):
        # Each split is a separate zip, so no split folder needs to be matched. The label comes only
        # from the class folder that directly contains the image, whatever the folders above it
        # are called (e.g. Train/Flood/img (1).jpg or data/Dataset_Images/Train/Flood/img (1).jpg).
        if split not in self.SOURCE["assets"]:
            raise ValueError(f"Unknown AIDERv2 split: {split!r}")

        with zipfile.ZipFile(data_path) as archive:
            # Sort for a deterministic example order across runs.
            for info in sorted(archive.infolist(), key=lambda i: i.filename):
                parts = info.filename.split("/")
                if (
                    info.is_dir()
                    or "__MACOSX" in parts
                    or parts[-1].startswith("._")
                    or not parts[-1].lower().endswith(_IMAGE_EXTENSIONS)
                ):
                    continue

                label = _FOLDER_TO_LABEL[parts[-2].lower()]  # KeyError on an unexpected folder: fail loudly

                image_bytes = archive.read(info)
                with PILImage.open(io.BytesIO(image_bytes)) as img:
                    # Keep the original bytes when already RGB; convert grayscale or CMYK frames.
                    image = image_bytes if img.mode == "RGB" else img.convert("RGB")

                yield info.filename, {"image": image, "label": label}
