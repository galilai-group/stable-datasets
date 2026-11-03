import io
import tarfile

from PIL import Image as PILImage

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import (
    BaseDatasetBuilder,
    _default_dest_folder,
    bulk_download,
)


# Breed names in label-index order. Official labels are 1-based and follow the
# sorted WordNet-ID folder names (e.g. "n02085620-Chihuahua" -> label 1).
_STANFORD_DOGS_CLASS_NAMES = [
    "Chihuahua", "Japanese_spaniel", "Maltese_dog", "Pekinese", "Shih-Tzu",
    "Blenheim_spaniel", "papillon", "toy_terrier", "Rhodesian_ridgeback", "Afghan_hound",
    "basset", "beagle", "bloodhound", "bluetick", "black-and-tan_coonhound",
    "Walker_hound", "English_foxhound", "redbone", "borzoi", "Irish_wolfhound",
    "Italian_greyhound", "whippet", "Ibizan_hound", "Norwegian_elkhound", "otterhound",
    "Saluki", "Scottish_deerhound", "Weimaraner", "Staffordshire_bullterrier", "American_Staffordshire_terrier",
    "Bedlington_terrier", "Border_terrier", "Kerry_blue_terrier", "Irish_terrier", "Norfolk_terrier",
    "Norwich_terrier", "Yorkshire_terrier", "wire-haired_fox_terrier", "Lakeland_terrier", "Sealyham_terrier",
    "Airedale", "cairn", "Australian_terrier", "Dandie_Dinmont", "Boston_bull",
    "miniature_schnauzer", "giant_schnauzer", "standard_schnauzer", "Scotch_terrier", "Tibetan_terrier",
    "silky_terrier", "soft-coated_wheaten_terrier", "West_Highland_white_terrier", "Lhasa", "flat-coated_retriever",
    "curly-coated_retriever", "golden_retriever", "Labrador_retriever", "Chesapeake_Bay_retriever", "German_short-haired_pointer",
    "vizsla", "English_setter", "Irish_setter", "Gordon_setter", "Brittany_spaniel",
    "clumber", "English_springer", "Welsh_springer_spaniel", "cocker_spaniel", "Sussex_spaniel",
    "Irish_water_spaniel", "kuvasz", "schipperke", "groenendael", "malinois",
    "briard", "kelpie", "komondor", "Old_English_sheepdog", "Shetland_sheepdog",
    "collie", "Border_collie", "Bouvier_des_Flandres", "Rottweiler", "German_shepherd",
    "Doberman", "miniature_pinscher", "Greater_Swiss_Mountain_dog", "Bernese_mountain_dog", "Appenzeller",
    "EntleBucher", "boxer", "bull_mastiff", "Tibetan_mastiff", "French_bulldog",
    "Great_Dane", "Saint_Bernard", "Eskimo_dog", "malamute", "Siberian_husky",
    "affenpinscher", "basenji", "pug", "Leonberg", "Newfoundland",
    "Great_Pyrenees", "Samoyed", "Pomeranian", "chow", "keeshond",
    "Brabancon_griffon", "Pembroke", "Cardigan", "toy_poodle", "miniature_poodle",
    "standard_poodle", "Mexican_hairless", "dingo", "dhole", "African_hunting_dog",
]  # fmt: skip

# Stanford URLs first. The fallbacks are Internet Archive (Wayback Machine) captures served raw
# (`id_`); their content digests match the original files, and the checksums make any copy that
# differs fail the download.
_STANFORD_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs"
_WAYBACK_URL = "https://web.archive.org/web/{timestamp}id_/" + _STANFORD_URL
_IMAGES_DOWNLOAD = DownloadInfo(
    url=f"{_STANFORD_URL}/images.tar",
    fallbacks=[_WAYBACK_URL.format(timestamp="20250724000413") + "/images.tar"],
    checksum="sha256:a443fc2f9a851aa1a7b8e4fa4af94b8459fce2c9f5a2f4dd4669cf9673a4ec58",
)
_LISTS_DOWNLOAD = DownloadInfo(
    url=f"{_STANFORD_URL}/lists.tar",
    fallbacks=[_WAYBACK_URL.format(timestamp="20241221112710") + "/lists.tar"],
    checksum="sha256:34b47cacd9a98b5d150e084f24d29391c084c55272295ec65c85651bc35f4d6c",
)


class StanfordDogs(BaseDatasetBuilder):
    """Fine-grained image classification.
    The `Stanford Dogs <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ dataset
    contains 20,580 RGB images of 120 dog breeds, built from ImageNet. The official
    split has 100 training images per class (12,000 total) and the remaining
    8,580 images for testing.
    """

    VERSION = Version("1.0.0")

    # Both splits come from the same image archive (downloaded once); the official
    # train/test membership comes from the separate list archive.
    SOURCE = DatasetSource(
        homepage="http://vision.stanford.edu/aditya86/ImageNetDogs/",
        assets={"train": _IMAGES_DOWNLOAD, "test": _IMAGES_DOWNLOAD},
        citation="""@inproceedings{khosla2011novel,
                         author    = {Aditya Khosla and Nityananda Jayadevaprakash and Bangpeng Yao and Li Fei-Fei},
                         title     = {Novel Dataset for Fine-Grained Image Categorization},
                         booktitle = {First Workshop on Fine-Grained Visual Categorization, IEEE Conference on Computer Vision and Pattern Recognition},
                         year      = {2011},
                         month     = {June},
                         address   = {Colorado Springs, CO}}""",
    )

    LISTS_DOWNLOAD = _LISTS_DOWNLOAD

    def _info(self):
        return DatasetInfo(
            description="""The Stanford Dogs dataset contains 20,580 images of 120 dog breeds from around the world, built from ImageNet for fine-grained image categorization. The official split uses 100 images per class for training (12,000 images) and the remaining 8,580 images for testing. See http://vision.stanford.edu/aditya86/ImageNetDogs/ for more information.""",
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=_STANFORD_DOGS_CLASS_NAMES),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _load_split_labels(self, split):
        """Return {"<wnid>-<breed>/<file>.jpg": 0-based label} for the official split."""
        from scipy.io import loadmat

        download_dir = getattr(self, "_raw_download_dir", None)
        if download_dir is None:
            download_dir = _default_dest_folder()
        lists_path = bulk_download([self.LISTS_DOWNLOAD], dest_folder=download_dir)[0]

        with tarfile.open(lists_path, "r") as archive:
            mat = loadmat(io.BytesIO(archive.extractfile(f"{split}_list.mat").read()))
        return {str(f[0][0]): int(label[0]) - 1 for f, label in zip(mat["file_list"], mat["labels"])}

    def _generate_examples(self, data_path, split):
        split_labels = self._load_split_labels(split)

        found = 0
        with tarfile.open(data_path, "r|") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                # Archive paths are "Images/<wnid>-<breed>/<file>.jpg"; split lists omit the prefix.
                rel_path = "/".join(member.name.split("/")[-2:])
                label = split_labels.get(rel_path)
                if label is None:
                    continue
                found += 1
                yield rel_path, {"image": _to_rgb_bytes(archive.extractfile(member).read()), "label": label}

        if found != len(split_labels):
            raise ValueError(f"Stanford Dogs {split} split lists {len(split_labels)} images but found {found}.")


def _to_rgb_bytes(image_bytes):
    """Return the image bytes unchanged if RGB, else re-encode as RGB JPEG.

    The source archive has one non-RGB image: n02105855_2933.jpg, which is actually an RGBA PNG.
    Left as is, its 4 channels would break batching with the 3-channel images.
    """
    with PILImage.open(io.BytesIO(image_bytes)) as img:
        if img.mode == "RGB":
            return image_bytes
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=95)
        return buf.getvalue()
