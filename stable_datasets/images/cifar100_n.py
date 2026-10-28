import pickle
import tarfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Image, Version
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, download

from .cifar100 import CIFAR100

_CITATION = """
@inproceedings{wei2022learning,
  title={Learning with Noisy Labels Revisited: A Study Using Real-World Human 
  Annotations},
  author={Jiaheng Wei and Zhaowei Zhu and Hao Cheng and Tongliang Liu and Gang 
  Niu and Yang Liu},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=TBWA6PLJZQm}
}
"""

_CLEAN_LABEL_FIELD = "clean_label"
_ANNOTATION_FIELDS = ("noisy_label",)

_CIFAR100N_FINE_TO_COARSE = {
    fine_label: coarse_idx
    for coarse_idx, fine_labels in enumerate(
        [
            ("beaver", "dolphin", "otter", "seal", "whale"),
            ("aquarium_fish", "flatfish", "ray", "shark", "trout"),
            ("orchid", "poppy", "rose", "sunflower", "tulip"),
            ("bottle", "bowl", "can", "cup", "plate"),
            ("apple", "mushroom", "orange", "pear", "sweet_pepper"),
            ("clock", "keyboard", "lamp", "telephone", "television"),
            ("bed", "chair", "couch", "table", "wardrobe"),
            ("bee", "beetle", "butterfly", "caterpillar", "cockroach"),
            ("bear", "leopard", "lion", "tiger", "wolf"),
            ("bridge", "castle", "house", "road", "skyscraper"),
            ("cloud", "forest", "mountain", "plain", "sea"),
            ("camel", "cattle", "chimpanzee", "elephant", "kangaroo"),
            ("fox", "porcupine", "possum", "raccoon", "skunk"),
            ("crab", "lobster", "snail", "spider", "worm"),
            ("baby", "boy", "girl", "man", "woman"),
            ("crocodile", "dinosaur", "lizard", "snake", "turtle"),
            ("hamster", "mouse", "rabbit", "shrew", "squirrel"),
            ("maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"),
            ("bicycle", "bus", "motorcycle", "pickup_truck", "train"),
            ("lawn_mower", "rocket", "streetcar", "tank", "tractor"),
        ]
    )
    for fine_label in fine_labels
}


def _load_annotations(path, field_names, expected_count, num_classes):
    try:
        annotations = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Compatibility with PyTorch versions before weights_only was added.
        annotations = torch.load(path, map_location="cpu")

    if not isinstance(annotations, Mapping):
        raise ValueError(f"CIFAR-N annotations at {path!s} must contain a mapping.")

    result = {}
    for field_name in field_names:
        if field_name not in annotations:
            raise ValueError(f"CIFAR-N annotation file {path!s} is missing required field {field_name!r}.")
        values = annotations[field_name]
        values = np.asarray(values)
        if values.ndim != 1:
            raise ValueError(f"CIFAR-N field {field_name!r} must be one-dimensional, got shape {values.shape}.")
        if len(values) != expected_count:
            raise ValueError(
                f"CIFAR-N field {field_name!r} has {len(values)} labels, but CIFAR has {expected_count} training images."
            )
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"CIFAR-N field {field_name!r} must contain integer labels, got {values.dtype}.")
        if np.any(values < 0) or np.any(values >= num_classes):
            raise ValueError(f"CIFAR-N field {field_name!r} contains labels outside [0, {num_classes}).")
        result[field_name] = values.astype(np.int64, copy=False)
    return result


def _verify_clean_labels(annotations, clean_labels):
    if not np.array_equal(annotations[_CLEAN_LABEL_FIELD], clean_labels):
        raise ValueError(
            "CIFAR-N clean_label annotations do not match the CIFAR training labels in Python batch order."
        )


class CIFAR100N(BaseDatasetBuilder):
    """CIFAR-100 training images with human-annotated CIFAR-100N labels."""

    VERSION = Version("1.0.1")

    SOURCE = DatasetSource(
        homepage="https://github.com/UCSC-REAL/cifar-10-100n",
        assets={
            "cifar": DownloadInfo(url="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"),
            "annotations": DownloadInfo(
                url="https://raw.githubusercontent.com/UCSC-REAL/cifar-10-100n/main/data/CIFAR-100_human.pt",
                filename="CIFAR-100_human.pt",
            ),
        },
        citation=_CITATION,
    )

    def _info(self):
        class_names = CIFAR100._fine_labels()
        coarse_names = CIFAR100._coarse_labels()
        return DatasetInfo(
            description=(
                "CIFAR-100N provides real-world, noisy human annotations for the CIFAR-100 training data (and no annotations on test data)."
                "Each image has a noisy label based on the human annotations on top of its ground truth fine and coarse labels "
                "present in CIFAR-100. Same as CIFAR-100, CIFAR-100N dataset contains 50,000 32x32 color training images "
                "and 10,000 test images, categorized into 100 classes, grouped into 20 superclasses."
            ),
            features=Features(
                {
                    "image": Image(),
                    "label": ClassLabel(names=class_names),
                    "coarse_label": ClassLabel(names=coarse_names),
                    "noisy_label": ClassLabel(names=class_names),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _candidate_splits(self):
        return [Split.TRAIN, Split.TEST]

    def _split_generators(self):
        assets = self._source()["assets"]
        download_dir = Path(self._raw_download_dir) / "cifar100n"
        paths = {
            name: download(
                self._normalize_download_info(assets[name], asset_name=name),
                dest_folder=download_dir,
            )
            for name in ("cifar", "annotations")
        }
        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "cifar_path": paths["cifar"],
                    "annotations_path": paths["annotations"],
                    "split": Split.TRAIN,
                },
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={
                    "cifar_path": paths["cifar"],
                    "split": Split.TEST,
                },
            )
        ]

    def _generate_examples(self, cifar_path, split, annotations_path=None):
        with tarfile.open(cifar_path, "r:gz") as tar:
            train = split == "train"
            split_file = "cifar-100-python/train" if train else "cifar-100-python/test"
            file = tar.extractfile(split_file).read()
            data = pickle.loads(file, encoding="latin1")
            images = data["data"].reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1))
            clean_labels = data["fine_labels"]
            coarse_labels = data["coarse_labels"]

            if train:
                annotations = _load_annotations(
                    annotations_path,
                    (_CLEAN_LABEL_FIELD, *_ANNOTATION_FIELDS),
                    len(images),
                    num_classes=100,
                )
                _verify_clean_labels(annotations, clean_labels)
            else:
                annotations = None

            for idx, (image, label) in enumerate(zip(images, clean_labels)):
                yield (
                    idx,
                    {
                        "image": image,
                        "label": int(label),
                        "coarse_label": int(coarse_labels[idx]),
                        **{
                            name: int(annotations[name][idx]) if annotations is not None else None
                            for name in _ANNOTATION_FIELDS
                        },
                    },
                )

