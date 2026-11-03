import io
import os
import tarfile

from PIL import Image as PILImage

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, bulk_download


class OxfordIIITPet(BaseDatasetBuilder):
    """
    Oxford-IIIT Pet Dataset

    Abstract
    A 37 category pet dataset with roughly 200 images per class. Images have
    large variations in scale, pose, and lighting. Each image is annotated
    with a breed label (species + breed); this builder only consumes the
    breed label, not the trimap/bounding-box annotations.
    """

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://www.robots.ox.ac.uk/~vgg/data/pets/",
        citation=r"""@inproceedings{parkhi12a,
                         title={Cats and Dogs},
                         author={Parkhi, O. M. and Vedaldi, A. and Zisserman, A. and Jawahar, C. V.},
                         booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
                         year={2012}}""",
        assets={
            "images": DownloadInfo(url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"),
            "annotations": DownloadInfo(url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"),
        },
        # Annotations are CC BY-SA 4.0; image copyright remains with the original owners.
        license="CC BY-SA 4.0",
    )

    def _info(self):
        return DatasetInfo(
            description="Oxford-IIIT Pet dataset with 37 breed categories.",
            features=Features(
                {
                    "image": ImageFeature(encode_format="JPEG"),
                    "label": ClassLabel(names=self._labels()),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license=self.SOURCE["license"],
            citation=self.SOURCE["citation"],
        )

    def _split_generators(self):
        """
        Download the images and annotations assets needed to generate
        each split. Both files are shared across splits: the annotation
        archive's trainval.txt / test.txt lists define which images
        belong to which split.
        """
        source = self._source()

        asset_keys = ["images", "annotations"]
        local_paths = bulk_download([source["assets"][key] for key in asset_keys], dest_folder=self._raw_download_dir)
        path_map = dict(zip(asset_keys, local_paths))

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"path_map": path_map, "split": "trainval"},
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={"path_map": path_map, "split": "test"},
            ),
        ]

    def _candidate_splits(self) -> list:
        # SOURCE assets are keyed by archive ("images"/"annotations"), not by split.
        return [Split.TRAIN, Split.TEST]

    def _generate_examples(self, path_map, split):
        images_path = path_map["images"]
        annotations_path = path_map["annotations"]

        list_file = "annotations/trainval.txt" if split == "trainval" else "annotations/test.txt"
        with tarfile.open(annotations_path, "r:gz") as ann_tar:
            f = ann_tar.extractfile(list_file)
            lines = f.read().decode("utf-8").splitlines()

        # Each non-comment line is "<image_id> <class_id> <species> <breed_id>",
        # where class_id is the official 1-based breed id (1..37). Labels are
        # class_id - 1, so they line up with the order of `_labels()`.
        id_to_label = {}
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            image_id, class_id = line.split()[:2]
            id_to_label[image_id] = int(class_id) - 1

        with tarfile.open(images_path, "r:gz") as img_tar:
            for member in img_tar:
                if not (member.isfile() and member.name.endswith(".jpg")):
                    continue
                image_id = os.path.basename(member.name)[: -len(".jpg")]
                if image_id not in id_to_label:
                    continue

                f = img_tar.extractfile(member)
                if f is None:
                    continue
                image = PILImage.open(io.BytesIO(f.read())).convert("RGB")

                yield image_id, {"image": image, "label": id_to_label[image_id]}

    @staticmethod
    def _labels():
        """37 breed names, indexed by official class id - 1 (annotations/list.txt CLASS-ID 1..37)."""
        return [
            "Abyssinian",
            "american_bulldog",
            "american_pit_bull_terrier",
            "basset_hound",
            "beagle",
            "Bengal",
            "Birman",
            "Bombay",
            "boxer",
            "British_Shorthair",
            "chihuahua",
            "Egyptian_Mau",
            "english_cocker_spaniel",
            "english_setter",
            "german_shorthaired",
            "great_pyrenees",
            "havanese",
            "japanese_chin",
            "keeshond",
            "leonberger",
            "Maine_Coon",
            "miniature_pinscher",
            "newfoundland",
            "Persian",
            "pomeranian",
            "pug",
            "Ragdoll",
            "Russian_Blue",
            "saint_bernard",
            "samoyed",
            "scottish_terrier",
            "shiba_inu",
            "Siamese",
            "Sphynx",
            "staffordshire_bull_terrier",
            "wheaten_terrier",
            "yorkshire_terrier",
        ]
