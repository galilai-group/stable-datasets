"""DeFungi images with deterministic, duplicate-linked source-group splits."""

import hashlib
import re
import zipfile
from collections import defaultdict

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Value, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


_URL = "https://archive.ics.uci.edu/static/public/773/defungi.zip"
_SHA256 = "02020311d1f8207f3aa3bf8b2303a5106463ac0bf35adde4ae042e8b2716ff6c"
_CLASSES = ("H1", "H2", "H3", "H5", "H6")
_FILENAME = re.compile(r"(H[12356])/(H[12356])_(\d+)[a-z]*_\d+\.jpg")
_SPLITS = ("train", "validation", "test")


def _scan_archive(archive):
    """Collect metadata and byte hashes without extracting archive paths."""
    records = []
    names = set()
    for member in sorted(archive.infolist(), key=lambda item: item.filename):
        if member.is_dir() or not member.filename.lower().endswith(".jpg"):
            continue
        name = member.filename
        match = _FILENAME.fullmatch(name)
        if match is None or match[1] != match[2]:
            raise ValueError(f"Unexpected DeFungi filename: {name}")
        if name in names:
            raise ValueError(f"Duplicate DeFungi archive member: {name}")
        names.add(name)
        label = _CLASSES.index(match[1])
        # Conservatively keep all letters and patches sharing a numeric id together.
        source_group = f"{match[1]}_{int(match[3])}"
        digest = hashlib.sha256(archive.read(member)).hexdigest()
        records.append((name, label, source_group, digest))
    if not records:
        raise ValueError("DeFungi archive contains no supported images.")
    return records


def _split_groups(records):
    """Merge duplicate-linked sources, then partition groups within each class."""
    parent = {source: source for _, _, source, _ in records}

    def find(source):
        root = source
        while parent[root] != root:
            root = parent[root]
        while parent[source] != source:
            next_source = parent[source]
            parent[source] = root
            source = next_source
        return root

    seen = {}
    for name, label, source, digest in records:
        if digest in seen:
            other_label, other_source = seen[digest]
            if label != other_label:
                raise ValueError(f"Identical DeFungi image has conflicting labels: {name}")
            first, second = sorted((find(source), find(other_source)))
            parent[second] = first
        else:
            seen[digest] = (label, source)

    by_class = defaultdict(set)
    for _, label, source, _ in records:
        by_class[label].add(find(source))

    assignments = {}
    for label, groups in sorted(by_class.items()):
        groups = sorted(groups, key=lambda group: hashlib.sha256(f"defungi-v1:42:{group}".encode()).digest())
        if len(groups) < 3:
            raise ValueError(f"DeFungi class {_CLASSES[label]} needs at least three independent groups.")
        n_train = min(max(1, int(0.70 * len(groups))), len(groups) - 2)
        n_validation = min(max(1, int(0.15 * len(groups))), len(groups) - n_train - 1)
        for index, group in enumerate(groups):
            split = "train" if index < n_train else "validation" if index < n_train + n_validation else "test"
            assignments[group] = split
    return {source: (find(source), assignments[find(source)]) for source in parent}


class DeFungi(BaseDatasetBuilder):
    """Load all 9,114 released RGB patches, preserving original class codes.

    A custom split holds out filename-derived source groups and links groups
    containing byte-identical images. Source groups are not patient identities.
    """

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="https://archive.ics.uci.edu/dataset/773/defungi",
        assets={split: DownloadInfo(url=_URL, checksum=f"sha256:{_SHA256}") for split in _SPLITS},
        license="CC BY 4.0",
        citation="""@misc{sopo2021defungi,
    title = {DeFungi: Direct Mycological Examination of Microscopic Fungi Images},
    author = {Pineda Sopo, Camilo Javier and Hajati, Farshid and Gheisari, Soheila},
    year = {2021},
    eprint = {2109.07322},
    archivePrefix = {arXiv},
    doi = {10.48550/arXiv.2109.07322}
}""",
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "DeFungi contains 9,114 RGB microscopy patches of size 500 by 500, "
                "labelled H1, H2, H3, H5 and H6. A deterministic custom split "
                "partitions duplicate-linked, filename-derived source groups "
                "within each class using approximate 70/15/15 group proportions. "
                "All released images, including duplicates within a split, are retained. "
                "The split is not an official protocol or a patient-disjoint split."
            ),
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=list(_CLASSES)),
                    "image_id": Value("string"),
                    "source_group": Value("string"),
                    "split_group": Value("string"),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license=self.SOURCE["license"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        if split not in _SPLITS:
            raise ValueError(f"Unknown DeFungi split: {split!r}")
        with zipfile.ZipFile(data_path) as archive:
            records = _scan_archive(archive)
            groups = _split_groups(records)
            for name, label, source, _ in records:
                split_group, assigned = groups[source]
                if assigned == split:
                    yield (
                        name,
                        {
                            "image": archive.read(name),
                            "label": label,
                            "image_id": name,
                            "source_group": source,
                            "split_group": split_group,
                        },
                    )
