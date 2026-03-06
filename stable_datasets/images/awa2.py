import zipfile

from PIL import Image as PILImage

from stable_datasets.schema import ClassLabel, DatasetInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, bulk_download


class AWA2(BaseDatasetBuilder):
    """
    The Animals with Attributes 2 (AwA2) dataset provides images across 50 animal classes, useful for attribute-based classification
    and zero-shot learning research. See https://cvml.ista.ac.at/AwA2/ for more information.
    """

    VERSION = Version("1.0.0")

    SOURCE = {
        "homepage": "https://cvml.ista.ac.at/AwA2/",
        "citation": """@ARTICLE{8413121,
                         author={Xian, Yongqin and Lampert, Christoph H. and Schiele, Bernt and Akata, Zeynep},
                         journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
                         title={Zero-Shot Learning—A Comprehensive Evaluation of the Good, the Bad and the Ugly},
                         year={2019},
                         volume={41},
                         number={9},
                         pages={2251-2265},
                         keywords={Semantics;Visualization;Task analysis;Training;Fish;Protocols;Learning systems;Generalized zero-shot learning;transductive learning;image classification;weakly-supervised learning},
                         doi={10.1109/TPAMI.2018.2857768}}""",
        "assets": {
            "test": "https://cvml.ista.ac.at/AwA2/AwA2-data.zip",
        },
    }

    def _info(self):
        return DatasetInfo(
            description="""The AWA2 dataset is an image classification dataset with images of 50 classes, primarily used in attribute-based image recognition research. See https://cvml.ista.ac.at/AwA2/ for more information.""",
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(
                        names=[
                            "antelope",
                            "grizzly+bear",
                            "killer+whale",
                            "beaver",
                            "dalmatian",
                            "persian+cat",
                            "horse",
                            "german+shepherd",
                            "blue+whale",
                            "siamese+cat",
                            "skunk",
                            "mole",
                            "tiger",
                            "hippopotamus",
                            "leopard",
                            "moose",
                            "spider+monkey",
                            "humpback+whale",
                            "elephant",
                            "gorilla",
                            "ox",
                            "fox",
                            "sheep",
                            "seal",
                            "chimpanzee",
                            "hamster",
                            "squirrel",
                            "rhinoceros",
                            "rabbit",
                            "bat",
                            "giraffe",
                            "wolf",
                            "chihuahua",
                            "rat",
                            "weasel",
                            "otter",
                            "buffalo",
                            "zebra",
                            "giant+panda",
                            "deer",
                            "bobcat",
                            "pig",
                            "lion",
                            "mouse",
                            "polar+bear",
                            "collie",
                            "walrus",
                            "raccoon",
                            "cow",
                            "dolphin",
                        ]
                    ),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _split_generators(self, dl_manager=None):
        source = self._source()
        urls = list(source["assets"].values())
        local_paths = bulk_download(urls, dest_folder=self._raw_download_dir)
        return [SplitGenerator(name=Split.TEST, gen_kwargs={"archive_path": local_paths[0]})]

    def _generate_examples(self, archive_path):
        with zipfile.ZipFile(archive_path, "r") as z:
            class_names = self.info.features["label"].names
            name_to_idx = {name: idx for idx, name in enumerate(class_names)}
            root_dir = "Animals_with_Attributes2/JPEGImages/"

            for image_path in z.namelist():
                if not image_path.endswith(".jpg"):
                    continue
                # Extract class name from path: "Animals_with_Attributes2/JPEGImages/<class>/image.jpg"
                rel = image_path[len(root_dir) :]
                slash = rel.find("/")
                if slash < 0:
                    continue
                class_name = rel[:slash]
                if class_name not in name_to_idx:
                    continue
                with z.open(image_path) as image_file:
                    image = PILImage.open(image_file).convert("RGB")
                    yield image_path, {"image": image, "label": name_to_idx[class_name]}
