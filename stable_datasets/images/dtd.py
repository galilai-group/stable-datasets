import tarfile
from io import BytesIO

import datasets
from PIL import Image
from tqdm import tqdm

from stable_datasets.utils import BaseDatasetBuilder


class DTD(BaseDatasetBuilder):
    """Describable Textures Dataset (DTD)

    DTD is a texture database, consisting of 5640 images, organized according to a list of 47 terms (categories)
    inspired from human perception. There are 120 images for each category. Image sizes range between 300x300 and
    640x640, and the images contain at least 90% of the surface representing the category attribute. The images were
    collected from Google and Flickr by entering our proposed attributes and related terms as search queries. The images
    were annotated using Amazon Mechanical Turk in several iterations. For each image we provide key attribute (main
    category) and a list of joint attributes.

    The data is split in three equal parts, in train, validation and test, 40 images per class, for each split. We
    provide the ground truth annotation for both key and joint attributes, as well as the 10 splits of the data we used
    for evaluation.
    """

    VERSION = datasets.Version("1.0.0")

    SOURCE = {
        "homepage": "https://www.robots.ox.ac.uk/~vgg/data/dtd/",
        "assets": {
            "train": "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz",
            "test": "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz",
            "val": "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz",
        },
        "citation": """@InProceedings{cimpoi14describing,
                    Author    = {M. Cimpoi and S. Maji and I. Kokkinos and S. Mohamed and and A. Vedaldi},
                    Title     = {Describing Textures in the Wild},
                    Booktitle = {Proceedings of the {IEEE} Conf. on Computer Vision and Pattern Recognition ({CVPR})},
                    Year      = {2014}}""",
    }

    def _info(self):
        return datasets.DatasetInfo(
            description="""Describing Textures in the Wild (DTD) is a dataset for texture classification.
                           It contains 5640 images organized into 47 categories.""",
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
        with tarfile.open(data_path, "r:gz") as tar:
            split_file = f"dtd/labels/{split}1.txt"
            file_names = self._read_split_file(tar, split_file)

            for idx, file_name in enumerate(tqdm(file_names, desc=f"Processing {split} split")):
                member = tar.getmember(f"dtd/images/{file_name}")
                file = tar.extractfile(member)
                image = Image.open(BytesIO(file.read())).convert("RGB")

                yield (
                    idx,
                    {
                        "image": image,
                        "label": file_name.split("/")[0],
                    },
                )

    def _read_split_file(self, tar, split_file):
        """Helper function to read split file from the tar archive."""
        split_content = tar.extractfile(split_file).read().decode("utf-8")
        return split_content.splitlines()

    @staticmethod
    def _labels():
        return [
            "banded",
            "blotchy",
            "braided",
            "bubbly",
            "bumpy",
            "chequered",
            "cobwebbed",
            "cracked",
            "crosshatched",
            "crystalline",
            "dotted",
            "fibrous",
            "flecked",
            "freckled",
            "frilly",
            "gauzy",
            "grid",
            "grooved",
            "honeycombed",
            "interlaced",
            "knitted",
            "lacelike",
            "lined",
            "marbled",
            "matted",
            "meshed",
            "paisley",
            "perforated",
            "pitted",
            "pleated",
            "polka-dotted",
            "porous",
            "potholed",
            "scaly",
            "smeared",
            "spiralled",
            "sprinkled",
            "stained",
            "stratified",
            "striped",
            "studded",
            "swirly",
            "veined",
            "waffled",
            "woven",
            "wrinkled",
            "zigzagged",
        ]
