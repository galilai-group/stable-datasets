import io
import json
import os
import tarfile

from PIL import Image as PILImage

from stable_datasets.schema import (
    ClassLabel,
    DatasetInfo,
    DatasetSource,
    DownloadInfo,
    Features,
    Value,
    Version,
)
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, bulk_download


# LILA hosts the dataset on Google Cloud Storage and Azure (identical files); Azure is the fallback mirror.
_GCS_URL = "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps"
_AZURE_URL = "https://lilawildlife.blob.core.windows.net/lila-wildlife/caltechcameratraps"

# Category ids of the annotation files, in label order (label 0 = opossum, ..., label 14 = rodent).
# "empty" (id 30) is excluded: it has no training images, so a classifier could never predict it.
_CATEGORIES = {
    1: "opossum",
    3: "raccoon",
    5: "squirrel",
    6: "bobcat",
    7: "skunk",
    8: "dog",
    9: "coyote",
    10: "rabbit",
    11: "bird",
    16: "cat",
    21: "badger",
    33: "car",
    34: "deer",
    51: "fox",
    99: "rodent",
}

# Library split name -> official annotation file ("cis" = camera locations seen in training, "trans" = unseen).
_SPLIT_FILES = {
    Split.TRAIN: "train",
    Split.VALIDATION: "cis_val",
    Split.TEST: "cis_test",
    "trans_validation": "trans_val",
    "trans_test": "trans_test",
}


def _single_label_images(annotations):
    """Map file_name -> (category_id, location) for images with exactly one known (non-empty) category."""
    categories = {}
    for annotation in annotations["annotations"]:
        categories.setdefault(annotation["image_id"], set()).add(annotation["category_id"])
    return {
        image["file_name"]: (next(iter(categories[image["id"]])), image["location"])
        for image in annotations["images"]
        if len(categories.get(image["id"], ())) == 1 and next(iter(categories[image["id"]])) in _CATEGORIES
    }


class CCT20(BaseDatasetBuilder):
    """Caltech Camera Traps (CCT20): camera-trap images of 15 animal and vehicle classes from 20 locations."""

    VERSION = Version("1.0.0")

    # Single source-of-truth for dataset provenance + download locations.
    SOURCE = DatasetSource(
        homepage="https://lila.science/datasets/caltech-camera-traps",
        assets={
            "images": DownloadInfo(
                url=f"{_GCS_URL}/eccv_18_all_images_sm.tar.gz",
                fallbacks=[f"{_AZURE_URL}/eccv_18_all_images_sm.tar.gz"],
                checksum="md5:8143c17aa2a12872b66f284ff211531f",
                filename="cct20_images_sm.tar.gz",
            ),
            "annotations": DownloadInfo(
                url=f"{_GCS_URL}/eccv_18_annotations.tar.gz",
                fallbacks=[f"{_AZURE_URL}/eccv_18_annotations.tar.gz"],
                checksum="md5:66a1f481b44aa1edadf75c9cfbd27aba",
                filename="cct20_annotations.tar.gz",
            ),
        },
        citation="""@inproceedings{beery2018recognition,
                        title = {Recognition in Terra Incognita},
                        author = {Beery, Sara and Van Horn, Grant and Perona, Pietro},
                        booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
                        year = {2018}
                        }""",
        license="CDLA-Permissive-1.0",
    )

    def _info(self):
        return DatasetInfo(
            description="""CCT20 is the camera-trap benchmark of "Recognition in Terra Incognita" (ECCV 2018): 57,864
            images (1024 pixels wide) from 20 camera locations in the Southwestern United States. Besides the training
            split, it has validation and test splits at the 10 training locations ("cis") and at 10 unseen locations
            ("trans"). Images labeled "empty" and the 78 images with more than one category are excluded, leaving 15
            classes.""",
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=list(_CATEGORIES.values())),
                    "location": Value("int32"),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license=self.SOURCE["license"],
            citation=self.SOURCE["citation"],
        )

    def _split_generators(self):
        # Step 1: Download every file in the SOURCE assets dictionary (skips files already on disk).
        assets = self._source()["assets"]
        names = list(assets)
        files = bulk_download(
            [self._normalize_download_info(assets[name], asset_name=name) for name in names],
            dest_folder=self._raw_download_dir,
        )
        paths = dict(zip(names, files))  # e.g. paths["images"] = local path of the images archive

        # Step 2: One order per split: same images + annotations, different official annotation file.
        return [
            SplitGenerator(
                name=split,
                gen_kwargs={
                    "images_path": paths["images"],
                    "annotations_path": paths["annotations"],
                    "annotation_file": annotation_file,
                },
            )
            for split, annotation_file in _SPLIT_FILES.items()
        ]

    def _generate_examples(self, images_path, annotations_path, annotation_file):
        with tarfile.open(annotations_path) as archive:
            member = f"eccv_18_annotation_files/{annotation_file}_annotations.json"
            keep = _single_label_images(json.load(archive.extractfile(member)))
        labels = {category_id: label for label, category_id in enumerate(_CATEGORIES)}

        # The images archive is a .tar.gz, which can only be read front to back ("r|gz" streams it once).
        with tarfile.open(images_path, "r|gz") as archive:
            for member in archive:
                filename = os.path.basename(member.name)
                if not member.isfile() or filename not in keep:
                    continue
                category_id, location = keep[filename]
                image_bytes = archive.extractfile(member).read()
                image = PILImage.open(io.BytesIO(image_bytes))
                yield (
                    filename,
                    {
                        # 551 night-time infrared images are grayscale; convert them so every image is RGB.
                        "image": image_bytes if image.mode == "RGB" else image.convert("RGB"),
                        "label": labels[category_id],
                        "location": location,
                    },
                )

    def _candidate_splits(self):
        return list(_SPLIT_FILES)
