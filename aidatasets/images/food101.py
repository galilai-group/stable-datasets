import datasets
import os


class Food101(datasets.GeneratorBasedBuilder):
    """A challenging data set of 101 food categories, with 101,000 images.
    For each class, 250 manually reviewed test images are provided as well as 750 training images. On purpose, the
    training images were not cleaned, and thus still contain some amount of noise. This comes mostly in the form of
    intense colors and sometimes wrong labels. All images were rescaled to have a maximum side length of 512 pixels.
    """

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="""This is the Food 101 dataset, also available from https://www.vision.ee.ethz.ch/datasets_extra/food-101/
            It contains images of food, organized by type of food. It was used in the Paper "Food-101 – Mining 
            Discriminative Components with Random Forests" by Lukas Bossard, Matthieu Guillaumin and Luc Van Gool. It's 
            a good (large dataset) for testing computer vision techniques.""",
            features=datasets.Features(
            {"image": datasets.Image(), "label": datasets.ClassLabel(names=self._labels())}),
            supervised_keys=("image", "label"),
            homepage="https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/",
            citation="""@inproceedings{bossard14,
                         title = {Food-101 -- Mining Discriminative Components with Random Forests},
                         author = {Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc},
                         booktitle = {European Conference on Computer Vision},
                         year = {2014}}"""
        )

    def _split_generators(self, dl_manager):
        archive = dl_manager.download_and_extract(
            "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
        )
        archive = str(archive)
        train = open(os.path.join(archive, "food-101", "meta", "train.txt")).read().splitlines()
        test = open(os.path.join(archive, "food-101", "meta", "test.txt")).read().splitlines()
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"root": archive, "archives": train},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"root": archive, "archives": test},
            ),
        ]

    def _generate_examples(self, root, archives):
        for key, name in enumerate(archives):
            image_path = os.path.join(root, "food-101", "images", f"{name}.jpg")
            yield key, {
                "image": image_path,
                "label": name.split("/")[0],
            }

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
