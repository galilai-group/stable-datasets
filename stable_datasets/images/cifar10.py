import pickle
import tarfile

import datasets
import numpy as np

from stable_datasets.utils import BaseDatasetBuilder


class CIFAR10(BaseDatasetBuilder):
    """Image classification.
    The `CIFAR-10 < https: // www.cs.toronto.edu/~kriz/cifar.html >`_ dataset
    was collected by Alex Krizhevsky, Vinod Nair, and Geoffrey
    Hinton. It consists of 60000 32x32 colour images in 10 classes, with
    6000 images per class. There are 50000 training images and 10000 test images.
    The dataset is divided into five training batches and one test batch,
    each with 10000 images. The test batch contains exactly 1000 randomly
    selected images from each class. The training batches contain the
    remaining images in random order, but some training batches may
    contain more images from one class than another. Between them, the
    training batches contain exactly 5000 images from each class.
    """

    VERSION = datasets.Version("1.0.0")

    # Single source-of-truth for dataset provenance + download locations.
    SOURCE = {
        "homepage": "https://www.cs.toronto.edu/~kriz/cifar.html",
        "assets": {
            "train": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
            "test": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        },
        "citation": """@article{krizhevsky2009learning,
                         title={Learning multiple layers of features from tiny images},
                         author={Krizhevsky, Alex and Hinton, Geoffrey and others},
                         year={2009},
                         publisher={Toronto, ON, Canada}}""",
    }

    def _info(self):
        return datasets.DatasetInfo(
            description="""The CIFAR-10 dataset is an image classification dataset of 50,000 32x32 color training images and 10,000 test images, labeled over 10 categories. See https://www.cs.toronto.edu/~kriz/cifar.html for more information.""",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(
                        names=[
                            "airplane",
                            "automobile",
                            "bird",
                            "cat",
                            "deer",
                            "dog",
                            "frog",
                            "horse",
                            "ship",
                            "truck",
                        ]
                    ),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        """Generate examples from the tar.gz archive."""
        with tarfile.open(data_path, "r:gz") as tar:
            train = split == "train"
            if train:
                train_images, train_labels = [], []
                for batch_idx in range(1, 6):
                    file = tar.extractfile(f"cifar-10-batches-py/data_batch_{batch_idx}").read()
                    data_dict = pickle.loads(file, encoding="latin1")
                    train_images.append(data_dict["data"].reshape((-1, 3, 32, 32)))
                    train_labels.extend(data_dict["labels"])

                train_images = np.concatenate(train_images, axis=0)
                train_images = np.transpose(train_images, (0, 2, 3, 1))
                for idx, (image, label) in enumerate(zip(train_images, train_labels)):
                    yield idx, {"image": image, "label": label}
            else:
                file = tar.extractfile("cifar-10-batches-py/test_batch").read()
                data_dict = pickle.loads(file, encoding="latin1")
                test_images = data_dict["data"].reshape((-1, 3, 32, 32))
                test_images = np.transpose(test_images, (0, 2, 3, 1))
                test_labels = data_dict["labels"]

                for idx, (image, label) in enumerate(zip(test_images, test_labels)):
                    yield idx, {"image": image, "label": label}
