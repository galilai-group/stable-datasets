from PIL import Image
import datasets
import zipfile
import os


class AWA2(datasets.GeneratorBasedBuilder):
    """
    The Animals with Attributes 2 (AwA2) dataset provides images across 50 animal classes, useful for attribute-based classification
    and zero-shot learning research. See https://cvml.ista.ac.at/AwA2/ for more information.
    """

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="""The AWA2 dataset is an image classification dataset with images of 50 classes, primarily used in attribute-based image recognition research. See https://cvml.ista.ac.at/AwA2/ for more information.""",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(names=['antelope', 'grizzly+bear', 'killer+whale', 'beaver',
                                                        'dalmatian', 'persian+cat', 'horse', 'german+shepherd',
                                                        'blue+whale', 'siamese+cat', 'skunk', 'mole', 'tiger',
                                                        'hippopotamus', 'leopard', 'moose', 'spider+monkey',
                                                        'humpback+whale', 'elephant', 'gorilla', 'ox', 'fox', 'sheep',
                                                        'seal', 'chimpanzee', 'hamster', 'squirrel', 'rhinoceros',
                                                        'rabbit', 'bat', 'giraffe', 'wolf', 'chihuahua', 'rat',
                                                        'weasel', 'otter', 'buffalo', 'zebra', 'giant+panda', 'deer',
                                                        'bobcat', 'pig', 'lion', 'mouse', 'polar+bear', 'collie',
                                                        'walrus', 'raccoon', 'cow', 'dolphin']),
                }
            ),
            supervised_keys=("image", "label"),
            homepage="https://cvml.ista.ac.at/AwA2/",
            citation="""@ARTICLE{8413121,
                         author={Xian, Yongqin and Lampert, Christoph H. and Schiele, Bernt and Akata, Zeynep},
                         journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
                         title={Zero-Shot Learning—A Comprehensive Evaluation of the Good, the Bad and the Ugly}, 
                         year={2019},
                         volume={41},
                         number={9},
                         pages={2251-2265},
                         keywords={Semantics;Visualization;Task analysis;Training;Fish;Protocols;Learning systems;Generalized zero-shot learning;transductive learning;image classification;weakly-supervised learning},
                         doi={10.1109/TPAMI.2018.2857768}}"""
        )

    def _split_generators(self, dl_manager):
        # Download the dataset
        archive_path = dl_manager.download({
            "data": "https://cvml.ista.ac.at/AwA2/AwA2-data.zip"
        })
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"archive_paths": archive_path}
            )
        ]

    def _generate_examples(self, archive_path):
        # Open the zip file
        with zipfile.ZipFile(archive_path, "r") as z:
            # Use the class names from DatasetInfo for consistent label order
            class_names = self._info().features["label"].names

            # Create a mapping from class name to label index based on DatasetInfo order
            label_mapping = {name: idx for idx, name in enumerate(class_names)}

            root_dir = "Animals_with_Attributes2/JPEGImages/"
            for class_name in class_names:
                class_dir = os.path.join(root_dir, class_name)

                # Iterate through each image in the class folder
                for image_path in z.namelist():
                    if image_path.startswith(class_dir) and image_path.endswith(".jpg"):
                        with z.open(image_path) as image_file:
                            image = Image.open(image_file).convert("RGB")
                            label = label_mapping[class_name]
                            yield image_path, {"image": image, "label": label}
