import random
import tarfile
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import PurePosixPath

from PIL import Image as PILImage
from tqdm import tqdm

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


# Caltech-101 has no official split. We follow the transfer-learning protocol of Kornblith et al. (2019), also used
# by SimCLR and BYOL: 30 random training images per class, all remaining images for testing. The seed fixes the
# draw so every user gets the same split.
SPLIT_SEED = 0
TRAIN_IMAGES_PER_CLASS = 30

# Clutter class shipped in the archive; excluded by torchvision and by the standard 101-class benchmark.
BACKGROUND_CLASS = "BACKGROUND_Google"

_URL = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1"
# MD5 published on the CaltechDATA record: 3138e1922a9193bfa496528edbbc45d0
_CHECKSUM = "sha256:331234750fc7f77520e50d9565e8b6907b03565e30320da1401130db08c61f91"


class Caltech101(BaseDatasetBuilder):
    """Caltech-101 object recognition dataset.

    Caltech-101 contains pictures of objects belonging to 101 categories, collected by Fei-Fei Li, Marco Andreetto,
    and Marc'Aurelio Ranzato in 2003. Most categories have about 40 to 800 images, and most images are roughly
    300x200 pixels. The archive also ships a ``BACKGROUND_Google`` clutter category, which is excluded here, as in
    torchvision and the standard benchmark, leaving 8677 images in 101 classes.

    The dataset has no official train/test split. This builder implements the protocol of Kornblith et al. (2019),
    used for linear evaluation in SimCLR and BYOL: for every class, file names are sorted, shuffled with a fixed
    seed (``SPLIT_SEED = 0``), and the first 30 images go to ``train`` while the remainder go to ``test``. This gives
    3030 training and 5647 test images. Grayscale images are converted to RGB.
    """

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://data.caltech.edu/records/mzrjq-6wc02",
        assets={
            "train": DownloadInfo(url=_URL, checksum=_CHECKSUM, filename="caltech-101.zip"),
            "test": DownloadInfo(url=_URL, checksum=_CHECKSUM, filename="caltech-101.zip"),
        },
        citation="""@article{fei2006one,
                    title     = {One-shot learning of object categories},
                    author    = {Fei-Fei, Li and Fergus, Rob and Perona, Pietro},
                    journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
                    volume    = {28},
                    number    = {4},
                    pages     = {594--611},
                    year      = {2006},
                    doi       = {10.1109/TPAMI.2006.79}}""",
        license="CC BY 4.0",
    )

    def _info(self):
        return DatasetInfo(
            description="""Caltech-101 is an object recognition dataset with 8677 images in 101 categories
                           (excluding the BACKGROUND_Google clutter class). Splits follow the standard transfer
                           protocol of 30 seeded random training images per class, with the rest for testing.""",
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=self._labels()),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
            license=self.SOURCE["license"],
        )

    def _generate_examples(self, data_path, split):
        images = self._read_images(data_path)
        split_files = self._split_files(images.keys())[split]

        for idx, (label, file_name) in enumerate(tqdm(split_files, desc=f"Processing {split} split")):
            image_bytes = images[(label, file_name)]
            image = PILImage.open(BytesIO(image_bytes))
            # Most source images are already RGB JPEGs; storing their bytes as-is keeps the cache at the size of the
            # original archive, and re-encoding would not change the decoded pixels anyway.
            yield (
                idx,
                {
                    "image": image_bytes if image.mode == "RGB" else image.convert("RGB"),
                    "label": label,
                },
            )

    @staticmethod
    def _read_images(data_path):
        """Read every image of the 101 classes into memory, keyed by ``(class_name, file_name)``."""
        images = {}
        with zipfile.ZipFile(data_path) as archive:
            with (
                archive.open(Caltech101._inner_tar_name(archive)) as inner,
                tarfile.open(fileobj=inner, mode="r|gz") as tar,
            ):
                for member in tar:
                    parts = PurePosixPath(member.name).parts
                    if not member.isfile() or len(parts) != 3 or not parts[2].lower().endswith(".jpg"):
                        continue
                    class_name, file_name = parts[1], parts[2]
                    if class_name == BACKGROUND_CLASS or file_name.startswith("."):
                        continue
                    images[(class_name, file_name)] = tar.extractfile(member).read()
        return images

    @staticmethod
    def _inner_tar_name(archive):
        """Locate ``101_ObjectCategories.tar.gz`` in the outer zip, ignoring macOS resource-fork entries."""
        candidates = [
            name
            for name in archive.namelist()
            if name.endswith("101_ObjectCategories.tar.gz")
            and not name.startswith("__MACOSX/")
            and not PurePosixPath(name).name.startswith(".")
        ]
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one 101_ObjectCategories.tar.gz in the archive, found {candidates}.")
        return candidates[0]

    @staticmethod
    def _split_files(keys):
        """Assign ``(class_name, file_name)`` keys to the seeded 30-per-class train split and the test split."""
        files_by_class = defaultdict(list)
        for class_name, file_name in keys:
            files_by_class[class_name].append(file_name)

        rng = random.Random(SPLIT_SEED)
        splits = {"train": [], "test": []}
        for class_name in sorted(files_by_class):
            file_names = sorted(files_by_class[class_name])
            if len(file_names) <= TRAIN_IMAGES_PER_CLASS:
                raise ValueError(
                    f"Class '{class_name}' has {len(file_names)} images; need more than {TRAIN_IMAGES_PER_CLASS} "
                    "to leave at least one test image."
                )
            rng.shuffle(file_names)
            splits["train"].extend((class_name, f) for f in sorted(file_names[:TRAIN_IMAGES_PER_CLASS]))
            splits["test"].extend((class_name, f) for f in sorted(file_names[TRAIN_IMAGES_PER_CLASS:]))
        return splits

    @staticmethod
    def _labels():
        return [
            "Faces",
            "Faces_easy",
            "Leopards",
            "Motorbikes",
            "accordion",
            "airplanes",
            "anchor",
            "ant",
            "barrel",
            "bass",
            "beaver",
            "binocular",
            "bonsai",
            "brain",
            "brontosaurus",
            "buddha",
            "butterfly",
            "camera",
            "cannon",
            "car_side",
            "ceiling_fan",
            "cellphone",
            "chair",
            "chandelier",
            "cougar_body",
            "cougar_face",
            "crab",
            "crayfish",
            "crocodile",
            "crocodile_head",
            "cup",
            "dalmatian",
            "dollar_bill",
            "dolphin",
            "dragonfly",
            "electric_guitar",
            "elephant",
            "emu",
            "euphonium",
            "ewer",
            "ferry",
            "flamingo",
            "flamingo_head",
            "garfield",
            "gerenuk",
            "gramophone",
            "grand_piano",
            "hawksbill",
            "headphone",
            "hedgehog",
            "helicopter",
            "ibis",
            "inline_skate",
            "joshua_tree",
            "kangaroo",
            "ketch",
            "lamp",
            "laptop",
            "llama",
            "lobster",
            "lotus",
            "mandolin",
            "mayfly",
            "menorah",
            "metronome",
            "minaret",
            "nautilus",
            "octopus",
            "okapi",
            "pagoda",
            "panda",
            "pigeon",
            "pizza",
            "platypus",
            "pyramid",
            "revolver",
            "rhino",
            "rooster",
            "saxophone",
            "schooner",
            "scissors",
            "scorpion",
            "sea_horse",
            "snoopy",
            "soccer_ball",
            "stapler",
            "starfish",
            "stegosaurus",
            "stop_sign",
            "strawberry",
            "sunflower",
            "tick",
            "trilobite",
            "umbrella",
            "watch",
            "water_lilly",
            "wheelchair",
            "wild_cat",
            "windsor_chair",
            "wrench",
            "yin_yang",
        ]
