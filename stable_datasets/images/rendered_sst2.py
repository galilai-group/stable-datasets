import tarfile

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


_URL = "https://openaipublic.azureedge.net/clip/data/rendered-sst2.tgz"
_CHECKSUM = "sha256:cae70e31ca3dd38b5b44046e112276d7a9b4b6761b067b21decfd0b5b6777d05"


class RenderedSST2(BaseDatasetBuilder):
    """Rendered SST2: sentiment classification of sentences rendered as images.

    Introduced with CLIP to measure optical character recognition ability. Each image is a 448x448 RGB
    rendering of a sentence from the Stanford Sentiment Treebank v2 (SST-2), labelled positive or negative.
    The official train/valid/test partitions are kept as released.
    """

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://github.com/openai/CLIP/blob/main/data/rendered-sst2.md",
        assets={
            "train": DownloadInfo(url=_URL, checksum=_CHECKSUM),
            "valid": DownloadInfo(url=_URL, checksum=_CHECKSUM),
            "test": DownloadInfo(url=_URL, checksum=_CHECKSUM),
        },
        citation="""@inproceedings{radford2021learning,
                title     = {Learning transferable visual models from natural language supervision},
                author    = {Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
                booktitle = {International conference on machine learning},
                pages     = {8748--8763},
                year      = {2021},
                organization = {PmLR} }
                @inproceedings{socher2013recursive,
                title     = {Recursive deep models for semantic compositionality over a sentiment treebank},
                author    = {Socher, Richard and Perelygin, Alex and Wu, Jean and Chuang, Jason and Manning, Christopher D and Ng, Andrew Y and Potts, Christopher},
                booktitle = {Proceedings of the 2013 conference on empirical methods in natural language processing},
                pages     = {1631--1642},
                year      = {2013} }
        """,
    )

    def _info(self):
        return DatasetInfo(
            description="""Rendered SST2 contains sentences from the Stanford Sentiment Treebank rendered as
                448x448 RGB images, labelled with binary sentiment (negative or positive). It evaluates whether a
                vision model can read text in an image well enough to classify its sentiment.""",
            features=Features({"image": ImageFeature(), "label": ClassLabel(names=self._labels())}),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        """Yield the PNG bytes under ``rendered-sst2/<split>/<label>/`` in the tar archive."""
        prefix = f"rendered-sst2/{split}/"
        with tarfile.open(data_path, "r:gz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.startswith(prefix) or not member.name.endswith(".png"):
                    continue
                label_name = member.name.split("/")[2]
                with archive.extractfile(member) as f:
                    image_bytes = f.read()
                yield member.name, {"image": image_bytes, "label": self.info.features["label"].str2int(label_name)}

    @staticmethod
    def _labels():
        return ["negative", "positive"]
