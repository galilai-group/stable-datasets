"""ETH-80 cropped256 images with a fixed, instance-disjoint split."""

import re
import tarfile
from collections import Counter
from io import BytesIO
from pathlib import PurePosixPath

from PIL import Image as PILImage

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Value, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


class ETH80(BaseDatasetBuilder):
    """Eight categories, ten objects per category and 41 views per object.

    This contribution defines the split (it is not an official benchmark):
    the first six objects train, the next two validation, and the last two
    test, ordered by numeric instance ID within each class.
    ``object_id`` is globally unique; ``instance_id`` is local to the class.
    Segmentation maps in the archive are excluded.
    """

    VERSION = Version("1.0.0")
    LABELS = ("apple", "car", "cow", "cup", "dog", "horse", "pear", "tomato")
    INSTANCE_IDS = {
        label: (1, 2, 3, 5, 6, 7, 9, 11, 12, 14) if label == "car" else tuple(range(1, 11)) for label in LABELS
    }
    SPLIT_SLICES = {"train": slice(0, 6), "validation": slice(6, 8), "test": slice(8, 10)}
    SOURCE = DatasetSource(
        homepage=(
            "https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/"
            "object-recognition-and-scene-understanding/"
            "analyzing-appearance-and-contour-based-methods-for-object-categorization"
        ),
        assets={
            split: DownloadInfo(
                url="https://datasets.d2.mpi-inf.mpg.de/eth80/eth80-cropped256.tgz",
                checksum="sha256:3552a8199bbf011bcc795364496430dbf936816f89b96ddb7a28af67cc6e6c95",
            )
            for split in ("train", "validation", "test")
        },
        citation="""@inproceedings{leibe2003analyzing,
    title={Analyzing Appearance and Contour Based Methods for Object Categorization},
    author={Leibe, Bastian and Schiele, Bernt},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
    year={2003}
}""",
        license="No explicit dataset license found on the provider page; consult the original provider for reuse terms.",
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "ETH-80 cropped256: 3,280 RGB images of 80 objects in eight classes. "
                "Contribution-defined instance split: first six train, next two validation, last two test per class, in numeric ID order."
            ),
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=list(self.LABELS)),
                    "object_id": Value("string"),
                    "instance_id": Value("int32"),
                    "view_id": Value("string"),
                    "image_id": Value("string"),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE.homepage,
            citation=self.SOURCE.citation,
            license=self.SOURCE.license,
        )

    def _generate_examples(self, data_path, split):
        if split not in self.SPLIT_SLICES:
            raise ValueError(f"Unknown ETH80 split: {split!r}")
        pattern = re.compile(r"([a-z]+)([1-9]\d*)-(\d{3}-\d{3})\.png")
        with tarfile.open(data_path, "r:gz") as archive:
            images = []
            seen = set()
            counts = Counter()
            for member in archive.getmembers():
                path = PurePosixPath(member.name)
                if not member.isfile() or path.suffix != ".png" or "maps" in path.parts:
                    continue
                match = pattern.fullmatch(path.name)
                if match is None:
                    raise ValueError(f"Malformed ETH80 image filename: {member.name}")
                label, instance, view = match.groups()
                object_id = f"{label}{instance}"
                if label not in self.LABELS or path.parts != ("eth80-cropped256", object_id, path.name):
                    raise ValueError(f"Unexpected ETH80 image path: {member.name}")
                if path.name in seen:
                    raise ValueError(f"Duplicate ETH80 image: {path.name}")
                seen.add(path.name)
                counts[object_id] += 1
                images.append((member, label, int(instance), view, object_id, path.name))

            expected = {f"{label}{instance}": 41 for label in self.LABELS for instance in self.INSTANCE_IDS[label]}
            if dict(counts) != expected:
                raise ValueError("Incomplete ETH80 archive: expected 80 objects with 41 images each.")

            for member, label, instance, view, object_id, image_id in sorted(images, key=lambda row: row[-1]):
                if instance not in self.INSTANCE_IDS[label][self.SPLIT_SLICES[split]]:
                    continue
                with archive.extractfile(member) as source, PILImage.open(BytesIO(source.read())) as image:
                    rgb = image.convert("RGB")
                yield (
                    image_id,
                    {
                        "image": rgb,
                        "label": label,
                        "object_id": object_id,
                        "instance_id": instance,
                        "view_id": view,
                        "image_id": image_id,
                    },
                )
