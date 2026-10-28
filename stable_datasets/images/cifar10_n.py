import pickle
import tarfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Image, Version
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, download

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

_CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

_CLEAN_LABEL_FIELD = "clean_label"
_ANNOTATION_FIELDS = (
    "aggre_label",
    "worse_label",
    "random_label1",
    "random_label2",
    "random_label3",
)


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


class CIFAR10N(BaseDatasetBuilder):
    """CIFAR-10 training images with human-annotated CIFAR-10N labels."""

    VERSION = Version("1.0.1")

    SOURCE = DatasetSource(
        homepage="https://github.com/UCSC-REAL/cifar-10-100n",
        assets={
            "cifar": DownloadInfo(url="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"),
            "annotations": DownloadInfo(
                url="https://raw.githubusercontent.com/UCSC-REAL/cifar-10-100n/main/data/CIFAR-10_human.pt",
                filename="CIFAR-10_human.pt",
            ),
        },
        citation=_CITATION,
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "CIFAR-10N provides real-world, noisy human annotations for the CIFAR-10 training data (and no annotations on test data). "
                "Each image has multiple human annotation labels in addition to its ground-truth label present in CIFAR-10. "
                "Same as CIFAR-10, CIFAR-10N contains 50,000 32x32 color training images and 10,000 test images, "
                "categorized into 10 classes."
            ),
            features=Features(
                {
                    "image": Image(),
                    "label": ClassLabel(names=_CIFAR10_CLASSES),
                    **{name: ClassLabel(names=_CIFAR10_CLASSES) for name in _ANNOTATION_FIELDS},
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
        download_dir = Path(self._raw_download_dir) / "cifar10n"
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
            train = split == Split.TRAIN
            if train:
                images, labels = [], []
                for batch_idx in range(1, 6):
                    member = tar.extractfile(f"cifar-10-batches-py/data_batch_{batch_idx}")
                    if member is None:
                        raise ValueError(f"CIFAR-10 archive is missing data_batch_{batch_idx}.")
                    data_dict = pickle.loads(member.read(), encoding="latin1")
                    images.append(data_dict["data"].reshape((-1, 3, 32, 32)))
                    labels.extend(data_dict["labels"])

                images = np.concatenate(images, axis=0)
                images = np.transpose(images, (0, 2, 3, 1))
                clean_labels = np.asarray(labels, dtype=np.int64)
                annotations = _load_annotations(
                    annotations_path,
                    (_CLEAN_LABEL_FIELD, *_ANNOTATION_FIELDS),
                    len(images),
                    num_classes=10,
                )
                _verify_clean_labels(annotations, clean_labels)
            else:
                member = tar.extractfile("cifar-10-batches-py/test_batch")
                if member is None:
                    raise ValueError("CIFAR-10 archive is missing test_batch.")
                data_dict = pickle.loads(member.read(), encoding="latin1")
                images = data_dict["data"].reshape((-1, 3, 32, 32))
                images = np.transpose(images, (0, 2, 3, 1))
                clean_labels = np.asarray(data_dict["labels"], dtype=np.int64)
                annotations = None

        for idx, (image, label) in enumerate(zip(images, clean_labels)):
            row = {
                "image": image,
                "label": int(label),
                **{
                    name: int(annotations[name][idx]) if annotations is not None else None
                    for name in _ANNOTATION_FIELDS
                },
            }
            yield idx, row
