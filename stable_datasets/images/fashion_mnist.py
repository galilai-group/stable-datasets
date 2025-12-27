import gzip

import datasets
import numpy as np

from stable_datasets.utils import BaseDatasetBuilder


class FashionMNIST(BaseDatasetBuilder):
    """Grayscale image classification.

    `Fashion-MNIST` is a dataset of Zalando's article images consisting of a training set of 60,000 examples
    and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.
    """

    VERSION = datasets.Version("1.0.0")

    # Single source-of-truth for dataset provenance + download locations.
    SOURCE = {
        "homepage": "https://github.com/zalandoresearch/fashion-mnist",
        "assets": {
            "train": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
            "test": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
        },
        "citation": """@article{xiao2017fashion,
                         title={Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms},
                         author={Xiao, Han and Rasul, Kashif and Vollgraf, Roland},
                         journal={arXiv preprint arXiv:1708.07747},
                         year={2017}}""",
    }

    def _info(self):
        return datasets.DatasetInfo(
            description="Fashion-MNIST is a dataset of Zalando's article images for image classification tasks.",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(
                        names=[
                            "T-shirt/top",
                            "Trouser",
                            "Pullover",
                            "Dress",
                            "Coat",
                            "Sandal",
                            "Shirt",
                            "Sneaker",
                            "Bag",
                            "Ankle boot",
                        ]
                    ),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        """Generate examples from the gzip archive."""
        # Fashion-MNIST has separate files for images and labels
        # The data_path will be the images file, we need to construct the labels path
        if split == "train":
            images_file = data_path
            labels_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz"
        else:  # test
            images_file = data_path
            labels_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz"

        # Download labels file
        from stable_datasets.utils import _default_dest_folder, bulk_download

        download_dir = getattr(self, "_raw_download_dir", None)
        if download_dir is None:
            download_dir = _default_dest_folder()
        labels_file = bulk_download([labels_url], dest_folder=download_dir)[0]

        with gzip.open(images_file, "rb") as img_path:
            images = np.frombuffer(img_path.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)
        with gzip.open(labels_file, "rb") as lbl_path:
            labels = np.frombuffer(lbl_path.read(), dtype=np.uint8, offset=8)

        for idx, (image, label) in enumerate(zip(images, labels)):
            yield idx, {"image": image, "label": label}
