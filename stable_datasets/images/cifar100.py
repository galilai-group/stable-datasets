import pickle
import tarfile

import datasets

from stable_datasets.utils import BaseDatasetBuilder


class CIFAR100(BaseDatasetBuilder):
    """CIFAR-100 dataset, a variant of CIFAR-10 with 100 classes."""

    VERSION = datasets.Version("1.0.0")

    # Single source-of-truth for dataset provenance + download locations.
    SOURCE = {
        "homepage": "https://www.cs.toronto.edu/~kriz/cifar.html",
        "assets": {
            "train": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
            "test": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
        },
        "citation": """@article{krizhevsky2009learning,
                         title={Learning multiple layers of features from tiny images},
                         author={Krizhevsky, Alex and Hinton, Geoffrey and others},
                         year={2009},
                         publisher={Toronto, ON, Canada}}""",
    }

    def _info(self):
        return datasets.DatasetInfo(
            description="""The CIFAR-100 dataset contains 50,000 32x32 color training images and 10,000 test images,
                           categorized into 100 classes, grouped into 20 superclasses. Each image has a 'fine' label
                           (the class it belongs to) and a 'coarse' label (the superclass it belongs to).""",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(names=self._fine_labels()),
                    "superclass": datasets.ClassLabel(names=self._coarse_labels()),
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
            split_file = "cifar-100-python/train" if train else "cifar-100-python/test"
            file = tar.extractfile(split_file).read()
            data = pickle.loads(file, encoding="latin1")
            images = data["data"].reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1))
            fine_labels = data["fine_labels"]
            coarse_labels = data["coarse_labels"]

            for idx, (image, fine_label, coarse_label) in enumerate(zip(images, fine_labels, coarse_labels)):
                yield (
                    idx,
                    {
                        "image": image,
                        "label": fine_label,
                        "superclass": coarse_label,
                    },
                )

    @staticmethod
    def _fine_labels():
        """Returns the list of CIFAR-100 fine labels (100 classes)."""
        return [
            "apple",
            "aquarium_fish",
            "baby",
            "bear",
            "beaver",
            "bed",
            "bee",
            "beetle",
            "bicycle",
            "bottle",
            "bowl",
            "boy",
            "bridge",
            "bus",
            "butterfly",
            "camel",
            "can",
            "castle",
            "caterpillar",
            "cattle",
            "chair",
            "chimpanzee",
            "clock",
            "cloud",
            "cockroach",
            "couch",
            "crab",
            "crocodile",
            "cup",
            "dinosaur",
            "dolphin",
            "elephant",
            "flatfish",
            "forest",
            "fox",
            "girl",
            "hamster",
            "house",
            "kangaroo",
            "keyboard",
            "lamp",
            "lawn_mower",
            "leopard",
            "lion",
            "lizard",
            "lobster",
            "man",
            "maple_tree",
            "motorcycle",
            "mountain",
            "mouse",
            "mushroom",
            "oak_tree",
            "orange",
            "orchid",
            "otter",
            "palm_tree",
            "pear",
            "pickup_truck",
            "pine_tree",
            "plain",
            "plate",
            "poppy",
            "porcupine",
            "possum",
            "rabbit",
            "raccoon",
            "ray",
            "road",
            "rocket",
            "rose",
            "sea",
            "seal",
            "shark",
            "shrew",
            "skunk",
            "skyscraper",
            "snail",
            "snake",
            "spider",
            "squirrel",
            "streetcar",
            "sunflower",
            "sweet_pepper",
            "table",
            "tank",
            "telephone",
            "television",
            "tiger",
            "tractor",
            "train",
            "trout",
            "tulip",
            "turtle",
            "wardrobe",
            "whale",
            "willow_tree",
            "wolf",
            "woman",
            "worm",
        ]

    @staticmethod
    def _coarse_labels():
        """Returns the list of CIFAR-100 coarse labels (20 superclasses)."""
        return [
            "aquatic_mammals",
            "fish",
            "flowers",
            "food_containers",
            "fruit_and_vegetables",
            "household_electrical_devices",
            "household_furniture",
            "insects",
            "large_carnivores",
            "large_man-made_outdoor_things",
            "large_natural_outdoor_scenes",
            "large_omnivores_and_herbivores",
            "medium_mammals",
            "non-insect_invertebrates",
            "people",
            "reptiles",
            "small_mammals",
            "trees",
            "vehicles_1",
            "vehicles_2",
        ]
