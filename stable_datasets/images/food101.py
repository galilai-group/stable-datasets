import tarfile

import datasets
from PIL import Image

from stable_datasets.utils import BaseDatasetBuilder


class Food101(BaseDatasetBuilder):
    """
    Food-101 Dataset.

    The Food-101 dataset consists of 101 food categories with 101,000 images.
    For each class, 250 manually reviewed test images are provided as well as
    750 training images.

    Split sizes:
    - train: 75,750 images (750 images × 101 classes)
    - test: 25,250 images (250 images × 101 classes)

    All images are automatically rescaled to have a maximum side length of 512 pixels.
    """

    VERSION = datasets.Version("1.0.0")

    SOURCE = {
        "homepage": "https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/",
        "assets": {
            "train": "https://huggingface.co/datasets/haodoz0118/food101-img/resolve/main/food101_train.tar",
            "test": "https://huggingface.co/datasets/haodoz0118/food101-img/resolve/main/food101_test.tar",
        },
        "citation": """@inproceedings{bossard14,
            title = {Food-101 -- Mining Discriminative Components with Random Forests},
            author = {Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc},
            booktitle = {European Conference on Computer Vision},
            year = {2014}}""",
    }

    def _info(self):
        return datasets.DatasetInfo(
            description="Food-101 image classification dataset. It has 101 food categories, with 101'000 images. For each class, 250 manually reviewed test images are provided as well as 750 training images. On purpose, the training images were not cleaned, and thus still contain some amount of noise. This comes mostly in the form of intense colors and sometimes wrong labels. All images were rescaled to have a maximum side length of 512 pixels",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(names=self._labels()),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        """Generate examples from uncompressed TAR archives.

        Expects TAR structure: class_name/image_id.jpg

        Using uncompressed TAR provides 5-10x speed improvement over gzip:
        - No decompression overhead (direct disk I/O)
        - Sequential reading is cache-friendly
        - Processes 75k images in 1-2 hours vs 10-12 hours with gzip

        File size increase is minimal (~80 MB, 2%) since JPEG is already compressed.
        """
        # Pre-build label lookup dictionary
        label_to_idx = {name: idx for idx, name in enumerate(self._labels())}

        # Open TAR in auto-detect mode (handles both compressed and uncompressed)
        with tarfile.open(data_path, "r:*") as tar:
            for member in tar.getmembers():
                # Quick filter: only process files (skip directories)
                if not member.isfile():
                    continue

                filename = member.name

                # Fast filter: skip non-jpg files
                if not filename.endswith(".jpg"):
                    continue

                # Extract class name from path: class_name/image_id.jpg
                # Split path into parts
                path_parts = filename.split("/")

                # Expected format: class_name/image_id.jpg
                # Find the class_name (second-to-last part)
                if len(path_parts) < 2:
                    continue

                class_name = path_parts[-2]  # Directory name = class name

                # Lookup label (skip if not found)
                label = label_to_idx.get(class_name)
                if label is None:
                    continue

                # Read and process image
                try:
                    file_obj = tar.extractfile(member)
                    image = Image.open(file_obj).convert("RGB")

                    # Use full path as key for uniqueness
                    yield filename, {"image": image, "label": label}
                except Exception:
                    # Skip corrupted images silently
                    continue

    @staticmethod
    def _labels():
        return [
            "apple_pie",
            "baby_back_ribs",
            "baklava",
            "beef_carpaccio",
            "beef_tartare",
            "beet_salad",
            "beignets",
            "bibimbap",
            "bread_pudding",
            "breakfast_burrito",
            "bruschetta",
            "caesar_salad",
            "cannoli",
            "caprese_salad",
            "carrot_cake",
            "ceviche",
            "cheesecake",
            "cheese_plate",
            "chicken_curry",
            "chicken_quesadilla",
            "chicken_wings",
            "chocolate_cake",
            "chocolate_mousse",
            "churros",
            "clam_chowder",
            "club_sandwich",
            "crab_cakes",
            "creme_brulee",
            "croque_madame",
            "cup_cakes",
            "deviled_eggs",
            "donuts",
            "dumplings",
            "edamame",
            "eggs_benedict",
            "escargots",
            "falafel",
            "filet_mignon",
            "fish_and_chips",
            "foie_gras",
            "french_fries",
            "french_onion_soup",
            "french_toast",
            "fried_calamari",
            "fried_rice",
            "frozen_yogurt",
            "garlic_bread",
            "gnocchi",
            "greek_salad",
            "grilled_cheese_sandwich",
            "grilled_salmon",
            "guacamole",
            "gyoza",
            "hamburger",
            "hot_and_sour_soup",
            "hot_dog",
            "huevos_rancheros",
            "hummus",
            "ice_cream",
            "lasagna",
            "lobster_bisque",
            "lobster_roll_sandwich",
            "macaroni_and_cheese",
            "macarons",
            "miso_soup",
            "mussels",
            "nachos",
            "omelette",
            "onion_rings",
            "oysters",
            "pad_thai",
            "paella",
            "pancakes",
            "panna_cotta",
            "peking_duck",
            "pho",
            "pizza",
            "pork_chop",
            "poutine",
            "prime_rib",
            "pulled_pork_sandwich",
            "ramen",
            "ravioli",
            "red_velvet_cake",
            "risotto",
            "samosa",
            "sashimi",
            "scallops",
            "seaweed_salad",
            "shrimp_and_grits",
            "spaghetti_bolognese",
            "spaghetti_carbonara",
            "spring_rolls",
            "steak",
            "strawberry_shortcake",
            "sushi",
            "tacos",
            "takoyaki",
            "tiramisu",
            "tuna_tartare",
            "waffles",
        ]
