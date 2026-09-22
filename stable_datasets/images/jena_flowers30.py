import csv
import io
import zipfile

from PIL import Image as PILImage

from stable_datasets.schema import BuilderConfig, ClassLabel, DatasetInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


class JenaFlowers30(BaseDatasetBuilder):
    """Jena Flowers 30: 1,479 images of 30 wild-flowering plant species.

    The ``all`` configuration exposes the complete collection as ``train`` for
    custom partitions, without an independent test set. Configurations ``fold1``,
    ``fold2``, and ``fold3`` preserve the three original 279/279/921
    train/validation/test partitions.
    Images retain their original resolution and are converted to RGB.
    """

    VERSION = Version("1.0.0")
    BUILDER_CONFIGS = [
        BuilderConfig(
            name="all",
            version=VERSION,
            description="The full collection, without an independent test split.",
        )
    ] + [
        BuilderConfig(
            name=f"fold{fold}",
            version=Version("1.0.0"),
            description=f"Official train/validation/test partition {fold}.",
        )
        for fold in range(1, 4)
    ]
    DEFAULT_CONFIG_NAME = "all"

    SOURCE = {
        "homepage": "https://doi.org/10.7910/DVN/QDHYST",
        "assets": {
            "train": "https://dataverse.harvard.edu/api/access/datafile/3003527?format=original",
        },
        "citation": r"""@article{seeland2017plant,
            author = {Seeland, Marco and Rzanny, Michael and Alaqraa, Nedal and W{\"a}ldchen, Jana and M{\"a}der, Patrick},
            title = {Plant Species Classification Using Flower Images -- A Comparative Study of Local Feature Representations},
            journal = {PLOS ONE},
            volume = {12},
            number = {2},
            pages = {e0170629},
            year = {2017},
            doi = {10.1371/journal.pone.0170629}
        }""",
    }

    def _source(self):
        if self.config.name == "all":
            return self.SOURCE
        return {
            **self.SOURCE,
            "assets": dict.fromkeys(("train", "validation", "test"), self.SOURCE["assets"]["train"]),
        }

    def _info(self):
        return DatasetInfo(
            description=(
                "Jena Flowers 30: 1,479 images of 30 wild-flowering plant species. "
                "Use all for the complete collection or fold1/fold2/fold3 for the "
                "official train/validation/test partitions."
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
            license="CC BY-NC 4.0",
        )

    def _generate_examples(self, data_path, split):
        with zipfile.ZipFile(data_path) as archive:
            labels = {}
            with archive.open("classlabels.txt") as label_file:
                reader = csv.reader(io.TextIOWrapper(label_file, encoding="utf-8"), skipinitialspace=True)
                for filename, class_id in reader:
                    label = int(class_id) - 1
                    if not 0 <= label < len(self._labels()):
                        raise ValueError(f"Invalid Jena Flowers 30 class ID {class_id!r} for {filename}.")
                    if filename in labels:
                        raise ValueError(f"Duplicate Jena Flowers 30 label for {filename}.")
                    labels[filename] = label
            if not labels:
                raise ValueError("No labels found in Jena Flowers 30 classlabels.txt.")

            filenames = labels if self.config.name == "all" else self._fold_memberships(archive, labels)[split]
            # The label file is not in filename order; join on names, not row positions.
            for filename in sorted(filenames):
                member = f"jpg/{filename}"
                with archive.open(member) as image_file, PILImage.open(image_file) as original:
                    image = original.convert("RGB")
                yield member, {"image": image, "label": labels[filename]}

    def _fold_memberships(self, archive, labels):
        fold = self.config.name.removeprefix("fold")
        memberships = {}
        assigned = set()
        for split, prefix in (("train", "trn"), ("validation", "val"), ("test", "tst")):
            member = f"{prefix}{fold}.txt"
            image_ids = archive.read(member).split()
            if not image_ids:
                raise ValueError(f"No image IDs found in Jena Flowers 30 {member}.")
            memberships[split] = set()
            for raw_id in image_ids:
                image_id = int(raw_id)
                filename = f"JFT_{image_id:05d}.jpg"
                if filename not in labels:
                    raise ValueError(f"Unknown image ID {image_id} in Jena Flowers 30 {member}.")
                if filename in memberships[split]:
                    raise ValueError(f"Duplicate image ID {image_id} in Jena Flowers 30 {member}.")
                if filename in assigned:
                    raise ValueError(f"Overlapping split membership for Jena Flowers 30 {filename}.")
                memberships[split].add(filename)
                assigned.add(filename)
        missing = set(labels) - assigned
        if missing:
            raise ValueError(f"Missing split membership in Jena Flowers 30 {self.config.name}: {sorted(missing)}.")
        return memberships

    @staticmethod
    def _labels():
        """Preserve the official 1-based class IDs as names for 0-based labels."""
        return [str(label) for label in range(1, 31)]
