import tarfile
from io import BytesIO

from PIL import Image as PILImage
from tqdm import tqdm

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, bulk_download


class LFW(BaseDatasetBuilder):
    """Labeled Faces in the Wild (LFW), face-verification pairs

    LFW is a database of more than 13,000 face photographs of 5,749 people, collected from the web, designed for
    studying unconstrained face recognition. This builder exposes the official face *verification* protocol: each
    example is a pair of face images (``image1``, ``image2``) and a label saying whether the two photos show the same
    person (``same``) or different people (``different``).

    Splits follow LFW's official development ("View 1") protocol: ``train`` contains the 2,200 pairs in
    ``pairsDevTrain.txt`` and ``test`` contains the 1,000 pairs in ``pairsDevTest.txt``. Both are balanced between
    matched and mismatched pairs. Images are the original (unaligned) 250x250 RGB JPEGs.

    The files are downloaded from the figshare mirrors that scikit-learn also uses.
    """

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="http://vis-www.cs.umass.edu/lfw/",
        assets={
            "images": DownloadInfo(url="https://ndownloader.figshare.com/files/5976018"),  # lfw.tgz
            "train_pairs": DownloadInfo(url="https://ndownloader.figshare.com/files/5976012"),  # pairsDevTrain.txt
            "test_pairs": DownloadInfo(url="https://ndownloader.figshare.com/files/5976009"),  # pairsDevTest.txt
        },
        citation="""@TechReport{LFWTech,
                    author      = {Gary B. Huang and Manu Ramesh and Tamara Berg and Erik Learned-Miller},
                    title       = {Labeled Faces in the Wild: A Database for Studying Face Recognition in
                                   Unconstrained Environments},
                    institution = {University of Massachusetts, Amherst},
                    year        = {2007},
                    number      = {07-49},
                    month       = {October}}""",
    )

    def _info(self):
        return DatasetInfo(
            description="""Labeled Faces in the Wild (LFW) face-verification pairs: two face photographs and a label
                           saying whether they show the same person.""",
            features=Features(
                {
                    "image1": ImageFeature(),
                    "image2": ImageFeature(),
                    "label": ClassLabel(names=["different", "same"]),
                }
            ),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _split_generators(self):
        source = self._source()
        assets = source["assets"]
        local_paths = bulk_download(list(assets.values()), dest_folder=self._raw_download_dir)
        path = dict(zip(assets.keys(), local_paths))

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"archive_path": path["images"], "pairs_path": path["train_pairs"]},
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={"archive_path": path["images"], "pairs_path": path["test_pairs"]},
            ),
        ]

    def _generate_examples(self, archive_path, pairs_path):
        pairs = self._read_pairs(pairs_path)
        needed = {key for first, second, _ in pairs for key in (first, second)}

        # One sequential pass over the archive: random access into a .tgz is slow.
        images = {}
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tqdm(tar, desc="Reading LFW archive"):
                if not member.isfile():
                    continue
                key = "/".join(member.name.split("/")[-2:])  # "Name/Name_0001.jpg"
                if key in needed:
                    images[key] = tar.extractfile(member).read()
                    if len(images) == len(needed):
                        break

        missing = needed - images.keys()
        if missing:
            raise FileNotFoundError(f"{len(missing)} images listed in the pairs file are missing, e.g. {sorted(missing)[:3]}")

        for idx, (first, second, label) in enumerate(pairs):
            yield (
                idx,
                {
                    "image1": PILImage.open(BytesIO(images[first])).convert("RGB"),
                    "image2": PILImage.open(BytesIO(images[second])).convert("RGB"),
                    "label": label,
                },
            )

    @staticmethod
    def _image_key(name, number):
        return f"{name}/{name}_{int(number):04d}.jpg"

    @classmethod
    def _read_pairs(cls, pairs_path):
        """Parse a LFW pairs file into a list of (image1_key, image2_key, label_name).

        Matched pairs are ``Name  i  j`` (3 fields); mismatched pairs are ``Name1  i  Name2  j`` (4 fields). Header
        lines (a pair count, or fold/size counts) have 1 or 2 fields and are skipped.
        """
        pairs = []
        with open(pairs_path, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 3:
                    name, i, j = parts
                    pairs.append((cls._image_key(name, i), cls._image_key(name, j), "same"))
                elif len(parts) == 4:
                    name1, i, name2, j = parts
                    pairs.append((cls._image_key(name1, i), cls._image_key(name2, j), "different"))
        return pairs