import gzip
import numpy as np
import datasets


class FashionMNIST(datasets.GeneratorBasedBuilder):
    """Grayscale image classification.

    `Fashion-MNIST` is a dataset of Zalando's article images consisting of a training set of 60,000 examples
    and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.
    """

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="Fashion-MNIST is a dataset of Zalando's article images for image classification tasks.",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(names=[
                        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
                    ])
                }
            ),
            supervised_keys=("image", "label"),
            homepage="https://github.com/zalandoresearch/fashion-mnist",
            license="MIT License",
            citation="""@article{xiao2017fashion,
                         title={Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms},
                         author={Xiao, Han and Rasul, Kashif and Vollgraf, Roland},
                         journal={arXiv preprint arXiv:1708.07747},
                         year={2017}}"""
        )

    def _split_generators(self, dl_manager):
        urls = {
            "train_images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
            "train_labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
            "test_images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
            "test_labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
        }
        downloaded_files = dl_manager.download(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "images_path": downloaded_files["train_images"],
                    "labels_path": downloaded_files["train_labels"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "images_path": downloaded_files["test_images"],
                    "labels_path": downloaded_files["test_labels"],
                },
            ),
        ]

    def _generate_examples(self, images_path, labels_path):
        with gzip.open(images_path, "rb") as img_path:
            images = np.frombuffer(img_path.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)
        with gzip.open(labels_path, "rb") as lbl_path:
            labels = np.frombuffer(lbl_path.read(), dtype=np.uint8, offset=8)

        for idx, (image, label) in enumerate(zip(images, labels)):
            yield idx, {"image": image, "label": label}
