import numpy as np
import datasets
from PIL import Image


class KMNIST(datasets.GeneratorBasedBuilder):
    """Kuzushiji-MNIST and Kuzushiji-49 datasets."""
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="kmnist", description="Kuzushiji-MNIST dataset with 10 classes."),
        datasets.BuilderConfig(name="k49mnist", description="Kuzushiji-49 dataset with 49 classes.")
    ]

    def _info(self):
        if self.config.name == "kmnist":
            num_classes = 10
        else:  # k49mnist
            num_classes = 49
        return datasets.DatasetInfo(
            description="Kuzushiji-MNIST and Kuzushiji-49 datasets.",
            features=datasets.Features({
                "image": datasets.Image(),  # Automatically converts to PIL.Image
                "label": datasets.ClassLabel(num_classes=num_classes),
            }),
            supervised_keys=("image", "label"),
            homepage="http://codh.rois.ac.jp/kmnist/",
            citation="""
                @online{clanuwat2018deep,
                  author       = {Tarin Clanuwat and Mikel Bober-Irizar and Asanobu Kitamoto and Alex Lamb and Kazuaki Yamamoto and David Ha},
                  title        = {Deep Learning for Classical Japanese Literature},
                  date         = {2018-12-03},
                  year         = {2018},
                  eprintclass  = {cs.CV},
                  eprinttype   = {arXiv},
                  eprint       = {cs.CV/1812.01718},
                }
            """
        )

    def _split_generators(self, dl_manager):
        urls = {
            "kmnist": {
                "train_imgs": "http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-imgs.npz",
                "train_labels": "http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-labels.npz",
                "test_imgs": "http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-imgs.npz",
                "test_labels": "http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-labels.npz",
            },
            "k49mnist": {
                "train_imgs": "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-imgs.npz",
                "train_labels": "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-labels.npz",
                "test_imgs": "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-imgs.npz",
                "test_labels": "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-labels.npz",
            },
        }
        selected_urls = urls[self.config.name]
        downloaded_files = dl_manager.download(selected_urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "images_path": downloaded_files["train_imgs"],
                    "labels_path": downloaded_files["train_labels"]
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "images_path": downloaded_files["test_imgs"],
                    "labels_path": downloaded_files["test_labels"]
                }
            ),
        ]

    def _generate_examples(self, images_path, labels_path):
        images = np.load(images_path)["arr_0"]
        labels = np.load(labels_path)["arr_0"]

        for idx, (image, label) in enumerate(zip(images, labels)):
            # Convert each image to a PIL.Image object
            image = Image.fromarray(image, mode="L")  # Mode "L" for grayscale images
            yield idx, {"image": image, "label": int(label)}
