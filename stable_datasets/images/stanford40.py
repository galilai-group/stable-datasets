"""Stanford 40 Actions with the official suggested train/test split."""

import io
import zipfile
from pathlib import PurePosixPath

from PIL import Image as PILImage

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Image, Version
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, bulk_download


_BASE_URL = "http://vision.stanford.edu/Datasets/"
# Internet Archive snapshots of the official files, byte-identical (SHA-256 below).
_ARCHIVE_URL = "https://web.archive.org/web/{}id_/" + _BASE_URL
# Fixed alphabetical mapping shared by training and test samples.
_CLASS_NAMES = (
    "applauding",
    "blowing_bubbles",
    "brushing_teeth",
    "cleaning_the_floor",
    "climbing",
    "cooking",
    "cutting_trees",
    "cutting_vegetables",
    "drinking",
    "feeding_a_horse",
    "fishing",
    "fixing_a_bike",
    "fixing_a_car",
    "gardening",
    "holding_an_umbrella",
    "jumping",
    "looking_through_a_microscope",
    "looking_through_a_telescope",
    "phoning",
    "playing_guitar",
    "playing_violin",
    "pouring_liquid",
    "pushing_a_cart",
    "reading",
    "riding_a_bike",
    "riding_a_horse",
    "rowing_a_boat",
    "running",
    "shooting_an_arrow",
    "smoking",
    "taking_photos",
    "texting_message",
    "throwing_frisby",
    "using_a_computer",
    "walking_the_dog",
    "washing_dishes",
    "watching_TV",
    "waving_hands",
    "writing_on_a_board",
    "writing_on_a_book",
)


def _to_rgb_bytes(data):
    """Return JPEG bytes unchanged if RGB; otherwise convert to RGB and encode losslessly as PNG."""
    with PILImage.open(io.BytesIO(data)) as image:
        if image.mode == "RGB":
            return data
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        return buffer.getvalue()


class Stanford40(BaseDatasetBuilder):
    """40 human actions in still images: 4,000 training and 5,532 test images.

    Images retain their full frame and original resolution. Bounding boxes
    are not used. Resize and normalize at training time. Class IDs follow
    the alphabetical order of the original action names, including spelling
    and capitalization. No additional validation split is created. All
    images are RGB: the few non-RGB source files (e.g. the grayscale
    ``feeding_a_horse_191.jpg``) are converted to RGB.
    """

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage=_BASE_URL + "40actions.html",
        citation="""@inproceedings{yao2011human,
  title={Human Action Recognition by Learning Bases of Action Attributes and Parts},
  author={Yao, Bangpeng and Jiang, Xiaoye and Khosla, Aditya and Lin, Andy Lai
          and Guibas, Leonidas and Fei-Fei, Li},
  booktitle={2011 International Conference on Computer Vision},
  year={2011}
}""",
        assets={
            "images": DownloadInfo(
                url=_BASE_URL + "Stanford40_JPEGImages.zip",
                fallbacks=[_ARCHIVE_URL.format("20240804122919") + "Stanford40_JPEGImages.zip"],
                checksum="sha256:a1a08cb27345e33680f3a33f9cf2214be4c0989b9b3365692ca5903b733ba2a9",
            ),
            "splits": DownloadInfo(
                url=_BASE_URL + "Stanford40_ImageSplits.zip",
                fallbacks=[_ARCHIVE_URL.format("20240804115240") + "Stanford40_ImageSplits.zip"],
                checksum="sha256:cc04534f632fd613ffeaff716018baa52ff471959c33179ef5d41bf6360474e5",
            ),
        },
        license="Unknown",
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "Stanford40 contains 9,532 still images of 40 human actions, "
                "using the official suggested split of 4,000 training and "
                "5,532 test images. Images are full-frame RGB, without bounding-box crops."
            ),
            features=Features({"image": Image(), "label": ClassLabel(names=list(_CLASS_NAMES))}),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license=self.SOURCE["license"],
            citation=self.SOURCE["citation"],
        )

    def _candidate_splits(self):
        # Download asset names are not dataset split names.
        return [Split.TRAIN, Split.TEST]

    def _split_generators(self):
        assets = self._source()["assets"]
        images_path, splits_path = bulk_download(
            [assets["images"], assets["splits"]], dest_folder=self._raw_download_dir
        )
        return [
            SplitGenerator(
                name=split_name,
                gen_kwargs={"images_path": images_path, "splits_path": splits_path, "split": split},
            )
            for split_name, split in ((Split.TRAIN, "train"), (Split.TEST, "test"))
        ]

    @staticmethod
    def _read_splits(archive):
        split_files = {}
        for split in ("train", "test"):
            names = archive.read(f"ImageSplits/{split}.txt").decode("utf-8-sig").splitlines()
            names = [name.strip() for name in names if name.strip()]
            if not names:
                raise ValueError(f"Empty Stanford40 {split} split")
            if len(names) != len(set(names)):
                raise ValueError(f"Duplicate filenames in Stanford40 {split} split")
            for name in names:
                if PurePosixPath(name).name != name or "\\" in name or not name.endswith(".jpg"):
                    raise ValueError(f"Invalid Stanford40 image filename: {name!r}")
                action, separator, index = name[:-4].rpartition("_")
                if not separator or action not in _CLASS_NAMES or not index.isdigit():
                    raise ValueError(f"Invalid Stanford40 action or image index: {name!r}")
            split_files[split] = names
        if set(split_files["train"]) & set(split_files["test"]):
            raise ValueError("Stanford40 training and test splits overlap")
        return split_files

    def _generate_examples(self, images_path, splits_path, split):
        if split not in ("train", "test"):
            raise ValueError(f"Unknown Stanford40 split: {split!r}")
        with zipfile.ZipFile(splits_path) as splits_archive:
            filenames = self._read_splits(splits_archive)[split]
        label_ids = {name: index for index, name in enumerate(_CLASS_NAMES)}
        with zipfile.ZipFile(images_path) as images_archive:
            for filename in filenames:
                action = filename[:-4].rsplit("_", 1)[0]
                yield (
                    filename,
                    {
                        "image": _to_rgb_bytes(images_archive.read(f"JPEGImages/{filename}")),
                        "label": label_ids[action],
                    },
                )
