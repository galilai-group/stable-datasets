from pathlib import Path

from stable_datasets.schema import ClassLabel, DatasetInfo, Features, Image, Version
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, download

from .imagenet_1k import _default_class_names, _ImageNetArchiveMixin


class ImageNet100(_ImageNetArchiveMixin, BaseDatasetBuilder):
    """ImageNet-100 built by taking the first 100 class TARs from ImageNet-1K train archive."""

    VERSION = Version("2.0.0")
    SOURCE = {
        "homepage": "https://www.image-net.org/challenges/LSVRC/2012/",
        "assets": {
            "train": "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar",
        },
        "citation": """@article{deng2009imagenet,
        title={ImageNet: A large-scale hierarchical image database},
        author={Deng, Jia and others},
        journal={CVPR},
        year={2009}
    }""",
    }

    def __init__(self, streaming: bool = True, **kwargs):
        self.streaming = streaming
        super().__init__(**kwargs)

    def _info(self):
        return DatasetInfo(
            description="ImageNet-100 subset generated from ImageNet-1K class archives.",
            features=Features({"image": Image(), "label": ClassLabel(names=_default_class_names(100))}),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _split_generators(self):
        train_path = download(self.SOURCE["assets"]["train"], dest_folder=self._raw_download_dir)
        return [SplitGenerator(name=Split.TRAIN, gen_kwargs={"data_path": train_path})]

    def _generate_examples(self, data_path, split=None):
        yield from self._iter_train_examples(Path(data_path), class_limit=100)
