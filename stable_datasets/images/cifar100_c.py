import numpy as np
import datasets
from PIL import Image
import tarfile
from io import BytesIO


class CIFAR100C(datasets.GeneratorBasedBuilder):
    """CIFAR-100-C dataset with corrupted CIFAR-100 images."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="""CIFAR-100-C is a corrupted version of the CIFAR-100 dataset, with 19 different types of 
                           corruptions applied to the images. The dataset consists of 100 classes and 5 levels 
                           of severity per corruption type.""",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(names=self._fine_labels()),
                    "corruption_name": datasets.Value("string"),
                    "corruption_level": datasets.Value("int32")
                }
            ),
            supervised_keys=("image", "label"),
            homepage="https://zenodo.org/records/3555552",
            license="CC BY 4.0",
            citation="""@article{hendrycks2019robustness,
                        title={Benchmarking Neural Network Robustness to Common Corruptions and Perturbations},
                        author={Dan Hendrycks and Thomas Dietterich},
                        journal={Proceedings of the International Conference on Learning Representations},
                        year={2019}}"""
        )

    def _split_generators(self, dl_manager):
        archive_path = dl_manager.download(
            "https://zenodo.org/records/3555552/files/CIFAR-100-C.tar?download=1"
        )
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"archive_path": archive_path, "corruptions": None},
            ),
        ]

    def _generate_examples(self, archive_path, corruptions=None):
        with tarfile.open(archive_path, "r") as tar:
            array_file = BytesIO()
            array_file.write(tar.extractfile("CIFAR-100-C/labels.npy").read())
            array_file.seek(0)
            labels = np.load(array_file)

            # Determine corruptions to load
            if isinstance(corruptions, str):
                corruptions = [corruptions]
            elif corruptions is None:
                corruptions = self._corruptions()
            images, labels_list, corruption_names, corruption_levels = [], [], [], []

            for corruption in corruptions:
                assert corruption in self._corruptions(), f"Unknown corruption type: {corruption}"
                print(f"Loading corruption: {corruption}")

                array_file = BytesIO()
                array_file.write(tar.extractfile(f"CIFAR-100-C/{corruption}.npy").read())
                array_file.seek(0)
                corrupted_images = np.load(array_file)

                for level in range(1, 6):
                    start_idx, end_idx = (level - 1) * 10000, level * 10000
                    images.extend(corrupted_images[start_idx:end_idx])
                    labels_list.extend(labels)
                    corruption_names.extend([corruption] * 10000)
                    corruption_levels.extend([level] * 10000)

            for idx, (image, label, corruption_name, corruption_level) in enumerate(
                    zip(images, labels_list, corruption_names, corruption_levels)):
                yield idx, {
                    "image": Image.fromarray(image),
                    "label": label,
                    "corruption_name": corruption_name,
                    "corruption_level": corruption_level,
                }

    @staticmethod
    def _corruptions():
        return [
            "zoom_blur", "speckle_noise", "spatter", "snow", "shot_noise", "saturate",
            "pixelate", "motion_blur", "jpeg_compression", "impulse_noise", "glass_blur",
            "gaussian_noise", "gaussian_blur", "frost", "fog", "elastic_transform",
            "defocus_blur", "contrast", "brightness"
        ]

    @staticmethod
    def _fine_labels():
        """Returns the list of CIFAR-100 fine labels (100 classes)."""
        return [
            "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
            "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
            "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup",
            "dinosaur", "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house",
            "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man",
            "maple_tree", "motorcycle", "mountain", "mouse", "mushroom", "oak_tree", "orange", "orchid",
            "otter", "palm_tree", "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
            "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew",
            "skunk", "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar", "sunflower",
            "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor", "train", "trout",
            "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
        ]
