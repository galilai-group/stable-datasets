import io
from zipfile import ZipFile

import datasets
from PIL import Image
from tqdm import tqdm

from stable_datasets.utils import BaseDatasetBuilder


class Food101(BaseDatasetBuilder):
    VERSION = datasets.Version("1.0.0")

    SOURCE = {
        "homepage": "https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/",
        "assets": {
            "train": "https://huggingface.co/datasets/haodoz0118/food101-img/resolve/main/food101_train.zip",
            "test": "https://huggingface.co/datasets/haodoz0118/food101-img/resolve/main/food101_test.zip",
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
        """Generate examples from the ZIP archives of images and labels."""
        labels = self._labels()
        label_to_idx = {name: idx for idx, name in enumerate(labels)}
        with ZipFile(data_path, "r") as archive:
            for entry in tqdm(archive.infolist(), desc=f"Processing {split} set"):
                if entry.filename.endswith(".jpg"):
                    content = archive.read(entry)
                    image = Image.open(io.BytesIO(content)).convert("RGB")

                    filename = entry.filename.split("/")[-1]
                    class_part = filename.split("_", 1)[1].rsplit(".", 1)[0]
                    label_name = class_part.lower().replace("-", "_").replace(".", "")
                    if label_name not in label_to_idx:
                        raise ValueError(f"Unknown label: {label_name}")

                    label = label_to_idx[label_name]

                    yield entry.filename, {"image": image, "label": label}

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
