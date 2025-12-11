import numpy as np
import datasets
import struct


class NotMNIST(datasets.GeneratorBasedBuilder):
    """NotMNIST Dataset that contains images of letters A-J."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="""A dataset that was created by Yaroslav Bulatov by taking some publicly available fonts and 
            extracting glyphs from them to make a dataset similar to MNIST. There are 10 classes, with letters A-J.""",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(num_classes=10),
                }
            ),
            supervised_keys=("image", "label"),
            homepage="https://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html",
            citation="""@misc{bulatov2011notmnist,
                          author={Yaroslav Bulatov},
                          title={notMNIST dataset},
                          year={2011},
                          url={http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html}
                        }""",
        )

    def _split_generators(self, dl_manager):
        # Download IDX files from the specified links
        urls = {
            "train_images": "https://github.com/davidflanagan/notMNIST-to-MNIST/raw/refs/heads/master/train-images-idx3-ubyte.gz",
            "train_labels": "https://github.com/davidflanagan/notMNIST-to-MNIST/raw/refs/heads/master/train-labels-idx1-ubyte.gz",
            "test_images": "https://github.com/davidflanagan/notMNIST-to-MNIST/raw/refs/heads/master/t10k-images-idx3-ubyte.gz",
            "test_labels": "https://github.com/davidflanagan/notMNIST-to-MNIST/raw/refs/heads/master/t10k-labels-idx1-ubyte.gz",
        }
        downloaded_files = dl_manager.download_and_extract(urls)

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
        # Read and parse the decompressed IDX files
        with open(images_path, "rb") as img_file:
            _, num_images, rows, cols = struct.unpack(">IIII", img_file.read(16))
            images = np.frombuffer(img_file.read(), dtype=np.uint8).reshape(num_images, rows, cols)

        with open(labels_path, "rb") as lbl_file:
            _, num_labels = struct.unpack(">II", lbl_file.read(8))
            labels = np.frombuffer(lbl_file.read(), dtype=np.uint8)

        assert len(images) == len(labels), "Mismatch between image and label counts."

        for idx, (image, label) in enumerate(zip(images, labels)):
            # Remove channel dimension for PIL compatibility
            yield idx, {"image": image, "label": label}
