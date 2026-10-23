"""Freiburg Groceries with the authors' five published train/test partitions."""

import tarfile
from io import BytesIO
from pathlib import PurePosixPath

from PIL import Image as PILImage

from stable_datasets.schema import (
    BuilderConfig,
    ClassLabel,
    DatasetInfo,
    DatasetSource,
    DownloadInfo,
    Features,
    Value,
    Version,
)
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.splits import SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, bulk_download


class FreiburgGroceries(BaseDatasetBuilder):
    """Load author partitions ``split0`` (default) through ``split4``.

    Training manifests contain repeated images, which are preserved in their
    original order. Use ``image_id`` to group duplicates before making an
    additional validation split. There is no official validation split.
    """

    VERSION = Version("1.0.0")
    BUILDER_CONFIGS = [BuilderConfig(name=f"split{i}") for i in range(5)]
    DEFAULT_CONFIG_NAME = "split0"
    LABELS = (
        "CANDY",
        "JUICE",
        "VINEGAR",
        "OIL",
        "CHOCOLATE",
        "PASTA",
        "RICE",
        "MILK",
        "SPICES",
        "HONEY",
        "JAM",
        "NUTS",
        "CHIPS",
        "SODA",
        "COFFEE",
        "BEANS",
        "TEA",
        "CORN",
        "CEREAL",
        "CAKE",
        "SUGAR",
        "WATER",
        "FLOUR",
        "TOMATO_SAUCE",
        "FISH",
    )
    SPLITS_REVISION = "8827d2f61c2838756a069562b1c2ff53568ae33b"
    SOURCE = DatasetSource(
        homepage="http://www2.informatik.uni-freiburg.de/~eitel/freiburg_groceries_dataset.html",
        assets={
            "images": DownloadInfo(
                url="http://aisdatasets.informatik.uni-freiburg.de/freiburg_groceries_dataset/freiburg_groceries_dataset.tar.gz",
                checksum="sha256:35c18d03bb6ed057f12a3c5aea0ac5eef9b8e1d5965d1e32142b7d1f1469e3f0",
            ),
            "annotations": DownloadInfo(
                url=f"https://codeload.github.com/PhilJd/freiburg_groceries_dataset/tar.gz/{SPLITS_REVISION}",
                filename="freiburg-groceries-splits.tar.gz",
                checksum="sha256:b07ba6e6c3760cb909f4b92de41f7ab505c3a910f8736613ea4b9712537aafb9",
            ),
        },
        citation="""@article{jund2016groceries,
    title={The Freiburg Groceries Dataset},
    author={Jund, Philipp and Abdo, Nichola and Eitel, Andreas and Burgard, Wolfram},
    journal={arXiv preprint arXiv:1611.05799},
    year={2016}
}""",
        license="Research purposes only; commercial use requires the authors' consent. Cite Jund et al. (2016).",
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "Freiburg Groceries: 4,947 released RGB images in 25 classes. "
                "Preserves author train/test manifests, including repeated training entries."
            ),
            features=Features(
                {"image": ImageFeature(), "label": ClassLabel(names=list(self.LABELS)), "image_id": Value("string")}
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE.homepage,
            citation=self.SOURCE.citation,
            license=self.SOURCE.license,
        )

    def _candidate_splits(self):
        return ["train", "test"]

    def _split_generators(self):
        images, annotations = bulk_download(self.SOURCE.assets.values(), dest_folder=self._raw_download_dir)
        partitions = self._read_partitions(annotations)
        return [
            SplitGenerator(name=split, gen_kwargs={"data_path": images, "rows": partitions[split]})
            for split in self._candidate_splits()
        ]

    def _read_partitions(self, annotation_path):
        prefix = f"freiburg_groceries_dataset-{self.SPLITS_REVISION}"
        fold = self.config.name.removeprefix("split")
        with tarfile.open(annotation_path, "r:*") as archive:

            def read_text(name):
                member = archive.getmember(f"{prefix}/{name}")
                if not member.isfile():
                    raise ValueError(f"Expected a regular annotation file: {name}")
                with archive.extractfile(member) as source:
                    return source.read().decode("utf-8")

            expected_labels = [f"{label} {i}" for i, label in enumerate(self.LABELS)]
            if read_text("classid.txt").splitlines() != expected_labels:
                raise ValueError("Freiburg classid.txt does not match the published label mapping.")
            partitions = {
                split: self._parse_manifest(read_text(f"splits/{split}{fold}.txt"))
                for split in self._candidate_splits()
            }
        train_ids = {path for path, _ in partitions["train"]}
        test_ids = {path for path, _ in partitions["test"]}
        if train_ids & test_ids:
            raise ValueError("Freiburg train/test manifests overlap.")
        if len(test_ids) != len(partitions["test"]):
            raise ValueError("Duplicate image in Freiburg test manifest.")
        return partitions

    @classmethod
    def _parse_manifest(cls, text):
        rows = []
        for line in text.splitlines():
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"Malformed Freiburg manifest line: {line!r}")
            path, label_text = fields
            label = int(label_text)
            parts = PurePosixPath(path).parts
            if (
                len(parts) != 2
                or not 0 <= label < len(cls.LABELS)
                or parts[0] != cls.LABELS[label]
                or not parts[1].startswith(parts[0])
                or not parts[1].endswith(".png")
                or str(PurePosixPath(path)) != path
            ):
                raise ValueError(f"Freiburg image path and label disagree: {line!r}")
            rows.append((path, label))
        if not rows:
            raise ValueError("Empty Freiburg split manifest.")
        return rows

    def _generate_examples(self, data_path, rows):
        # The official .tar.gz asset is actually an uncompressed tar archive.
        with tarfile.open(data_path, "r:*") as archive:
            members = {}
            for member in archive.getmembers():
                if member.name in members:
                    raise ValueError(f"Duplicate Freiburg archive member: {member.name}")
                members[member.name] = member
            for index, (image_id, label) in enumerate(rows):
                member = members.get(f"images/{image_id}")
                if member is None or not member.isfile():
                    raise ValueError(f"Missing regular Freiburg image: {image_id}")
                with archive.extractfile(member) as source, PILImage.open(BytesIO(source.read())) as image:
                    rgb = image.convert("RGB")
                yield index, {"image": rgb, "label": label, "image_id": image_id}
