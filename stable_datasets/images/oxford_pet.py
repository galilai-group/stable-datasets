import io
import os
import tarfile

from PIL import Image as PILImage
from tqdm import tqdm

from stable_datasets.schema import ClassLabel, DatasetInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, bulk_download


class OxfordPet(BaseDatasetBuilder):
    """Oxford-IIIT Pet Dataset

    The Oxford-IIIT Pet dataset is a 37-category pet image dataset with roughly 200 images per class,
    covering 12 cat breeds and 25 dog breeds. Images exhibit large variations in scale, pose, and
    lighting. Each image is annotated with a breed label, a head bounding box, and a pixel-level
    trimap segmentation mask. The dataset ships with an official train/test split used in the original
    paper: 3,680 training images and 3,669 test images, for a total of 7,349 images.
    """

    VERSION = Version("1.0.0")

    SOURCE = {
        "homepage": "https://www.robots.ox.ac.uk/~vgg/data/pets/",
        "assets": {
            "images": "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz",
            "annotations": "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz",
        },
        "citation": """@InProceedings{parkhi12a,
            author       = "Omkar M. Parkhi and Andrea Vedaldi and Andrew Zisserman and C. V. Jawahar",
            title        = "Cats and Dogs",
            booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition",
            year         = "2012",
        }""",
    }

    def _info(self):
        return DatasetInfo(
            description=(
                "The Oxford-IIIT Pet dataset contains 7,349 images covering 37 pet breeds "
                "(12 cats, 25 dogs) with roughly 200 images per class. The official split "
                "provides 3,680 training images and 3,669 test images."
            ),
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=self._labels()),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _split_generators(self):
        """Override default splitting because both the image tar and annotation tar
        are needed to build any split."""
        source = self._source()

        key_url_map = {
            "images": source["assets"]["images"],
            "annotations": source["assets"]["annotations"],
        }

        urls = list(key_url_map.values())
        local_paths = bulk_download(urls, dest_folder=self._raw_download_dir)
        path_map = dict(zip(key_url_map.keys(), local_paths))

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"path_map": path_map, "split": "train"},
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={"path_map": path_map, "split": "test"},
            ),
        ]

    def _generate_examples(self, path_map, split):
        """Yield (key, example) pairs for the requested split by joining split list with image tar."""
        images_path = path_map["images"]
        annotations_path = path_map["annotations"]

        split_file = "trainval.txt" if split == "train" else "test.txt"

        name_to_label = {}
        with tarfile.open(annotations_path, "r:gz") as tar:
            member = tar.getmember(f"annotations/{split_file}")
            with tar.extractfile(member) as f:
                for line in f.read().decode("utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    image_name, class_id = parts[0], int(parts[1])
                    name_to_label[image_name] = class_id - 1

        with tarfile.open(images_path, "r:gz") as tar:
            for entry in tqdm(tar, desc=f"Processing {split} set"):
                if not entry.isfile() or not entry.name.endswith(".jpg"):
                    continue
                stem = os.path.basename(entry.name).rsplit(".", 1)[0]
                if stem not in name_to_label:
                    continue
                with tar.extractfile(entry) as f:
                    image = PILImage.open(io.BytesIO(f.read())).convert("RGB")
                    yield stem, {"image": image, "label": name_to_label[stem]}

    @staticmethod
    def _labels():
        """37 breeds in the canonical CLASS-ID order from annotations/list.txt (1-indexed there, 0-indexed here)."""
        return [
            "abyssinian",
            "american_bulldog",
            "american_pit_bull_terrier",
            "basset_hound",
            "beagle",
            "bengal",
            "birman",
            "bombay",
            "boxer",
            "british_shorthair",
            "chihuahua",
            "egyptian_mau",
            "english_cocker_spaniel",
            "english_setter",
            "german_shorthaired",
            "great_pyrenees",
            "havanese",
            "japanese_chin",
            "keeshond",
            "leonberger",
            "maine_coon",
            "miniature_pinscher",
            "newfoundland",
            "persian",
            "pomeranian",
            "pug",
            "ragdoll",
            "russian_blue",
            "saint_bernard",
            "samoyed",
            "scottish_terrier",
            "shiba_inu",
            "siamese",
            "sphynx",
            "staffordshire_bull_terrier",
            "wheaten_terrier",
            "yorkshire_terrier",
        ]
