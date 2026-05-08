import io
import tarfile

import scipy.io
from PIL import Image as PILImage
from tqdm import tqdm

from stable_datasets.schema import ClassLabel, DatasetInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, bulk_download


class StanfordDogs(BaseDatasetBuilder):
    """Stanford Dogs Dataset

    The Stanford Dogs dataset contains 20,580 images of 120 dog breeds, built on top of ImageNet.
    Each breed has roughly 150-250 images and an associated synset id (e.g. ``n02085620-Chihuahua``).
    The dataset ships with an official train/test split (12,000 train, 8,580 test) defined in
    ``train_list.mat`` and ``test_list.mat``.
    """

    VERSION = Version("1.0.0")

    SOURCE = {
        "homepage": "http://vision.stanford.edu/aditya86/ImageNetDogs/",
        "assets": {
            "images": "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar",
            "lists": "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar",
        },
        "citation": """@inproceedings{KhoslaYaoJayadevaprakashFeiFei_FGVC2011,
            author    = {Aditya Khosla and Nityananda Jayadevaprakash and Bangpeng Yao and Li Fei-Fei},
            title     = {Novel Dataset for Fine-Grained Image Categorization},
            booktitle = {First Workshop on Fine-Grained Visual Categorization, IEEE Conference on Computer Vision and Pattern Recognition},
            year      = {2011},
            month     = {June},
            address   = {Colorado Springs, CO}
        }""",
    }

    def _info(self):
        return DatasetInfo(
            description=(
                "Stanford Dogs: 20,580 images of 120 dog breeds drawn from ImageNet, with an "
                "official 12,000 / 8,580 train/test split."
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
        """Download both images.tar and lists.tar; each split joins them via the .mat split files."""
        source = self._source()
        key_url_map = {
            "images": source["assets"]["images"],
            "lists": source["assets"]["lists"],
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
        """Read the .mat split file from lists.tar, then iterate images.tar once and yield the matched entries."""
        images_path = path_map["images"]
        lists_path = path_map["lists"]

        split_mat_name = "train_list.mat" if split == "train" else "test_list.mat"
        with tarfile.open(lists_path, "r") as lists_tar:
            with lists_tar.extractfile(split_mat_name) as f:
                mat = scipy.io.loadmat(io.BytesIO(f.read()))

        # mat 'file_list' entries look like 'n02085620-Chihuahua/n02085620_5927.jpg' (relative to Images/)
        # Labels in the mat are 1-indexed; convert to 0-indexed for the ClassLabel feature.
        rel_files = [str(x[0][0]) for x in mat["file_list"]]
        raw_labels = mat["labels"].ravel().tolist()
        archive_path_to_label = {f"Images/{rel}": int(lab) - 1 for rel, lab in zip(rel_files, raw_labels)}

        with tarfile.open(images_path, "r") as img_tar:
            for entry in tqdm(img_tar, desc=f"Processing {split} set"):
                if not entry.isfile() or not entry.name.endswith(".jpg"):
                    continue
                label = archive_path_to_label.get(entry.name)
                if label is None:
                    continue
                with img_tar.extractfile(entry) as f:
                    image = PILImage.open(io.BytesIO(f.read())).convert("RGB")
                yield entry.name, {"image": image, "label": label}

    @staticmethod
    def _labels():
        """120 breed names in canonical CLASS-ID order from file_list.mat (1-indexed there, 0-indexed here)."""
        return [
            "chihuahua",
            "japanese_spaniel",
            "maltese_dog",
            "pekinese",
            "shih-tzu",
            "blenheim_spaniel",
            "papillon",
            "toy_terrier",
            "rhodesian_ridgeback",
            "afghan_hound",
            "basset",
            "beagle",
            "bloodhound",
            "bluetick",
            "black-and-tan_coonhound",
            "walker_hound",
            "english_foxhound",
            "redbone",
            "borzoi",
            "irish_wolfhound",
            "italian_greyhound",
            "whippet",
            "ibizan_hound",
            "norwegian_elkhound",
            "otterhound",
            "saluki",
            "scottish_deerhound",
            "weimaraner",
            "staffordshire_bullterrier",
            "american_staffordshire_terrier",
            "bedlington_terrier",
            "border_terrier",
            "kerry_blue_terrier",
            "irish_terrier",
            "norfolk_terrier",
            "norwich_terrier",
            "yorkshire_terrier",
            "wire-haired_fox_terrier",
            "lakeland_terrier",
            "sealyham_terrier",
            "airedale",
            "cairn",
            "australian_terrier",
            "dandie_dinmont",
            "boston_bull",
            "miniature_schnauzer",
            "giant_schnauzer",
            "standard_schnauzer",
            "scotch_terrier",
            "tibetan_terrier",
            "silky_terrier",
            "soft-coated_wheaten_terrier",
            "west_highland_white_terrier",
            "lhasa",
            "flat-coated_retriever",
            "curly-coated_retriever",
            "golden_retriever",
            "labrador_retriever",
            "chesapeake_bay_retriever",
            "german_short-haired_pointer",
            "vizsla",
            "english_setter",
            "irish_setter",
            "gordon_setter",
            "brittany_spaniel",
            "clumber",
            "english_springer",
            "welsh_springer_spaniel",
            "cocker_spaniel",
            "sussex_spaniel",
            "irish_water_spaniel",
            "kuvasz",
            "schipperke",
            "groenendael",
            "malinois",
            "briard",
            "kelpie",
            "komondor",
            "old_english_sheepdog",
            "shetland_sheepdog",
            "collie",
            "border_collie",
            "bouvier_des_flandres",
            "rottweiler",
            "german_shepherd",
            "doberman",
            "miniature_pinscher",
            "greater_swiss_mountain_dog",
            "bernese_mountain_dog",
            "appenzeller",
            "entlebucher",
            "boxer",
            "bull_mastiff",
            "tibetan_mastiff",
            "french_bulldog",
            "great_dane",
            "saint_bernard",
            "eskimo_dog",
            "malamute",
            "siberian_husky",
            "affenpinscher",
            "basenji",
            "pug",
            "leonberg",
            "newfoundland",
            "great_pyrenees",
            "samoyed",
            "pomeranian",
            "chow",
            "keeshond",
            "brabancon_griffon",
            "pembroke",
            "cardigan",
            "toy_poodle",
            "miniature_poodle",
            "standard_poodle",
            "mexican_hairless",
            "dingo",
            "dhole",
            "african_hunting_dog",
        ]
