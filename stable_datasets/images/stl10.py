import tarfile
import numpy as np
import datasets


class STL10(datasets.GeneratorBasedBuilder):
    """STL-10 Dataset
    The STL-10 dataset is a dataset for developing unsupervised feature learning,
    deep learning, and self-taught learning algorithms.
    """

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="STL-10 dataset for unsupervised feature learning. "
                        "Includes labeled and unlabeled images.",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(
                        names=[
                            "airplane", "bird", "car", "cat", "deer",
                            "dog", "horse", "monkey", "ship", "truck"
                        ]
                    ),
                }
            ),
            supervised_keys=("image", "label"),
            homepage="https://cs.stanford.edu/~acoates/stl10/",
            citation="""@article{coates2011analysis,
                         title={An analysis of single-layer networks in unsupervised feature learning},
                         author={Coates, Adam and Ng, Andrew Y},
                         journal={AISTATS},
                         year={2011}}""",
        )

    def _split_generators(self, dl_manager):
        archive_path = dl_manager.download(
            "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
        )
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"archive_path": archive_path, "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"archive_path": archive_path, "split": "test"},
            ),
            datasets.SplitGenerator(
                name="unlabeled",
                gen_kwargs={"archive_path": archive_path, "split": "unlabeled"},
            ),
        ]

    def _generate_examples(self, archive_path, split):
        with tarfile.open(archive_path, "r:gz") as tar:
            if split == "train":
                images_file = "stl10_binary/train_X.bin"
                labels_file = "stl10_binary/train_y.bin"
            elif split == "test":
                images_file = "stl10_binary/test_X.bin"
                labels_file = "stl10_binary/test_y.bin"
            else:  # unlabeled
                images_file = "stl10_binary/unlabeled_X.bin"
                labels_file = None

            images = tar.extractfile(images_file).read()
            images = (
                np.frombuffer(images, dtype=np.uint8)
                .reshape(-1, 3, 96, 96)
                .transpose((0, 2, 3, 1))
            )

            if labels_file:
                labels = tar.extractfile(labels_file).read()
                labels = np.frombuffer(labels, dtype=np.uint8) - 1
                for idx, (image, label) in enumerate(zip(images, labels)):
                    yield idx, {"image": image, "label": label}
            else:
                for idx, image in enumerate(images):
                    yield idx, {"image": image, "label": -1}
