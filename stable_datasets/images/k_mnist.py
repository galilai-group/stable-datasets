import datasets
import numpy as np

from stable_datasets.utils import (
    BaseDatasetBuilder,
    _default_dest_folder,
    bulk_download,
)


class KMNIST(BaseDatasetBuilder):
    """Image classification.
    The `Kuzushiji-MNIST <http://codh.rois.ac.jp/kmnist/>`_ dataset consists
    of 70,000 28x28 grayscale images of 10 classes of Kuzushiji (cursive
    Japanese) characters, with 7,000 images per class. There are 60,000
    training images and 10,000 test images. Kuzushiji-MNIST is a drop-in
    replacement for the MNIST dataset, providing a more challenging
    alternative for benchmarking machine learning algorithms.
    """

    VERSION = datasets.Version("1.0.0")

    # Single source-of-truth for dataset provenance + download locations.
    SOURCE = {
        "homepage": "http://codh.rois.ac.jp/kmnist/",
        "assets": {
            "train": "https://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-imgs.npz",
            "test": "https://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-imgs.npz",
        },
        "citation": """@online{clanuwat2018deep,
                         author       = {Tarin Clanuwat and Mikel Bober-Irizar and Asanobu Kitamoto and Alex Lamb and Kazuaki Yamamoto and David Ha},
                         title        = {Deep Learning for Classical Japanese Literature},
                         date         = {2018-12-03},
                         year         = {2018},
                         eprintclass  = {cs.CV},
                         eprinttype   = {arXiv},
                         eprint       = {cs.CV/1812.01718}}""",
    }

    def _info(self):
        return datasets.DatasetInfo(
            description="""The Kuzushiji-MNIST dataset is an image classification dataset of 60,000 28x28 grayscale training images and 10,000 test images, labeled over 10 classes of cursive Japanese characters. See http://codh.rois.ac.jp/kmnist/ for more information.""",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(
                        names=[
                            "o",
                            "ki",
                            "su",
                            "tsu",
                            "na",
                            "ha",
                            "ma",
                            "ya",
                            "re",
                            "wo",
                        ]
                    ),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        """Generate examples from the npz archive."""
        data = np.load(data_path, allow_pickle=True)
        images = data["arr_0"]

        # Load labels from separate file
        label_url = self.SOURCE["assets"][split].replace("-imgs.npz", "-labels.npz")
        download_dir = getattr(self, "_raw_download_dir", None)
        if download_dir is None:
            download_dir = _default_dest_folder()

        # Download label file
        label_paths = bulk_download([label_url], dest_folder=download_dir)
        label_path = label_paths[0]
        labels = np.load(label_path, allow_pickle=True)["arr_0"]

        for idx, (image, label) in enumerate(zip(images, labels)):
            yield idx, {"image": image, "label": int(label)}
