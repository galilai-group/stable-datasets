import numpy as np

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Image, Version
from stable_datasets.utils import (
    BaseDatasetBuilder,
    _default_dest_folder,
    bulk_download,
)


# Romanized names of the 49 classes, in label-index order of the official
# k49_classmap.csv (U+3042 "a" ... U+3093 "n", then U+309D, the hiragana iteration mark).
_K49_CLASS_NAMES = [
    "a", "i", "u", "e", "o",
    "ka", "ki", "ku", "ke", "ko",
    "sa", "shi", "su", "se", "so",
    "ta", "chi", "tsu", "te", "to",
    "na", "ni", "nu", "ne", "no",
    "ha", "hi", "fu", "he", "ho",
    "ma", "mi", "mu", "me", "mo",
    "ya", "yu", "yo",
    "ra", "ri", "ru", "re", "ro",
    "wa", "wi", "we", "wo",
    "n",
    "iteration_mark",
]  # fmt: skip


class KMNIST49(BaseDatasetBuilder):
    """Image classification.
    The `Kuzushiji-49 <http://codh.rois.ac.jp/kmnist/>`_ dataset consists
    of 270,912 28x28 grayscale images of 49 classes of Kuzushiji (cursive
    Japanese) hiragana characters. There are 232,365 training images and
    38,547 test images. Unlike Kuzushiji-MNIST, the classes are highly
    imbalanced, ranging from a few hundred to several thousand images per class.
    """

    VERSION = Version("1.0.0")

    # Single source-of-truth for dataset provenance + download locations.
    SOURCE = DatasetSource(
        homepage="http://codh.rois.ac.jp/kmnist/",
        assets={
            "train": DownloadInfo(url="https://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-imgs.npz"),
            "test": DownloadInfo(url="https://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-imgs.npz"),
        },
        citation="""@online{clanuwat2018deep,
                         author       = {Tarin Clanuwat and Mikel Bober-Irizar and Asanobu Kitamoto and Alex Lamb and Kazuaki Yamamoto and David Ha},
                         title        = {Deep Learning for Classical Japanese Literature},
                         date         = {2018-12-03},
                         year         = {2018},
                         eprintclass  = {cs.CV},
                         eprinttype   = {arXiv},
                         eprint       = {cs.CV/1812.01718}}""",
    )

    # Labels ship as separate files, one per split.
    LABEL_URLS = {
        "train": "https://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-labels.npz",
        "test": "https://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-labels.npz",
    }

    def _info(self):
        return DatasetInfo(
            description="""The Kuzushiji-49 dataset is an image classification dataset of 232,365 28x28 grayscale training images and 38,547 test images, labeled over 49 imbalanced classes of cursive Japanese hiragana characters. See http://codh.rois.ac.jp/kmnist/ for more information. "KMNIST Dataset" (created by CODH), adapted from "Kuzushiji Dataset" (created by NIJL and others), doi:10.20676/00000341.""",
            features=Features(
                {
                    "image": Image(),
                    "label": ClassLabel(names=_K49_CLASS_NAMES),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license="CC BY-SA 4.0",
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        """Generate examples from the image and label npz archives."""
        images = np.load(data_path)["arr_0"]

        download_dir = getattr(self, "_raw_download_dir", None)
        if download_dir is None:
            download_dir = _default_dest_folder()
        label_path = bulk_download([self.LABEL_URLS[split]], dest_folder=download_dir)[0]
        labels = np.load(label_path)["arr_0"]

        if len(images) != len(labels):
            raise ValueError(f"K-49 {split} split has {len(images)} images but {len(labels)} labels.")

        for idx, (image, label) in enumerate(zip(images, labels)):
            yield idx, {"image": image, "label": int(label)}
