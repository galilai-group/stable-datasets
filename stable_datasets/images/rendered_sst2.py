import tarfile

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


class RenderedSST2(BaseDatasetBuilder):
    """Stanford Sentiment Treebank v2 sentences rendered as images for binary sentiment classification."""

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://github.com/openai/CLIP/blob/main/data/rendered-sst2.md",
        assets={
            "train": DownloadInfo(
                url="https://openaipublic.azureedge.net/clip/data/rendered-sst2.tgz",
                checksum="sha256:cae70e31ca3dd38b5b44046e112276d7a9b4b6761b067b21decfd0b5b6777d05",
            ),
            "validation": DownloadInfo(
                url="https://openaipublic.azureedge.net/clip/data/rendered-sst2.tgz",
                checksum="sha256:cae70e31ca3dd38b5b44046e112276d7a9b4b6761b067b21decfd0b5b6777d05",
            ),
            "test": DownloadInfo(
                url="https://openaipublic.azureedge.net/clip/data/rendered-sst2.tgz",
                checksum="sha256:cae70e31ca3dd38b5b44046e112276d7a9b4b6761b067b21decfd0b5b6777d05",
            ),
        },
        citation="""@inproceedings{radford2021learning,
    title={Learning Transferable Visual Models From Natural Language Supervision},
    author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and Krueger, Gretchen and Sutskever, Ilya},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2021}}
@inproceedings{socher2013recursive,
    title={Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank},
    author={Socher, Richard and Perelygin, Alex and Wu, Jean and Chuang, Jason and Manning, Christopher D. and Ng, Andrew and Potts, Christopher},
    booktitle={Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    year={2013}}""",
    )

    def _info(self):
        return DatasetInfo(
            description="""Rendered SST2 contains sentences from the Stanford Sentiment Treebank v2 rendered as
                448x448 RGB images by OpenAI to evaluate CLIP's OCR ability. The task is binary sentiment
                classification from images of text, with 6,920 training, 872 validation, and 1,821 test images.""",
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=["negative", "positive"]),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license="Unknown",
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        folder = {"train": "train", "validation": "valid", "test": "test"}[split]
        label_feature = self.info.features["label"]
        with tarfile.open(data_path, "r|gz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".png"):
                    continue
                parts = member.name.split("/")
                if len(parts) != 4 or parts[0] != "rendered-sst2" or parts[1] != folder:
                    continue
                if parts[2] not in label_feature.names:
                    continue
                label = label_feature.str2int(parts[2])
                with archive.extractfile(member) as image_file:
                    image_bytes = image_file.read()
                yield member.name, {"image": image_bytes, "label": label}
