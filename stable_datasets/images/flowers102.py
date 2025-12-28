import io
import os
import tarfile

import datasets
import scipy.io
from PIL import Image

from stable_datasets.utils import BaseDatasetBuilder, bulk_download


class Flowers102(BaseDatasetBuilder):
    """
    Flowers102 Dataset

    Abstract
    The Flowers102 dataset is a fine-grained image classification benchmark consisting of 102 flower categories commonly found in the United Kingdom. It was created to address the challenge of classifying objects with large intra-class variability and small inter-class differences. Each category contains between 40 and 258 images, totaling 8,189 images.

    Context
    Fine-grained visual categorization (FGVC) focuses on differentiating between similar sub-categories of objects (e.g., different species of flowers or birds). Flowers102 serves as a standard benchmark in this domain. Unlike general object recognition (e.g., CIFAR-10), where classes are visually distinct (car vs. dog), Flowers102 requires models to learn subtle features like petal shape, texture, and color patterns.

    Content
    The dataset consists of:
    - **Images:** 8,189 images stored in a single archive.
    - **Labels:** A MATLAB file mapping each image to one of 102 classes (0-101).
    - **Splits:** A predefined split ID file dividing the data into Training (1,020 images), Validation (1,020 images), and Test (6,149 images).
    """

    VERSION = datasets.Version("1.0.0")

    SOURCE = {
        "homepage": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/",
        "citation": r"""@inproceedings{nilsback2008flowers102,
                         title={Automated flower classification over a large number of classes},
                         author={Nilsback, Maria-Elena and Zisserman, Andrew},
                         booktitle={2008 Sixth Indian conference on computer vision, graphics \& image processing},
                         pages={722--729},
                         year={2008},
                         organization={IEEE}}""",
        "assets": {
            "images": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz",
            "labels": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat",
            "setid": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat",
        },
    }

    def _info(self):
        return datasets.DatasetInfo(
            description="Flowers102 dataset with 102 classes.",
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

    def _split_generators(self, dl_manager):
        """
        Override default splitting because we need all 3 files (images, labels, IDs)
        to generate examples for any split.
        """
        source = self._source()

        key_url_map = {
            "images": source["assets"]["images"],
            "labels": source["assets"]["labels"],
            "setid": source["assets"]["setid"],
        }

        urls = list(key_url_map.values())
        local_paths = bulk_download(urls, dest_folder=self._raw_download_dir)

        path_map = dict(zip(key_url_map.keys(), local_paths))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "path_map": path_map,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "path_map": path_map,
                    "split": "valid",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "path_map": path_map,
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, path_map, split):
        images_path = path_map["images"]
        labels_path = path_map["labels"]
        setid_path = path_map["setid"]

        labels_data = scipy.io.loadmat(labels_path)["labels"][0]
        labels_data = labels_data - 1

        setid_data = scipy.io.loadmat(setid_path)
        if split == "train":
            ids = setid_data["trnid"][0]
        elif split == "valid":
            ids = setid_data["valid"][0]
        else:
            ids = setid_data["tstid"][0]

        ids_set = set(ids)

        with tarfile.open(images_path, "r:gz") as tar:
            for member in tar:
                if member.isfile() and member.name.endswith(".jpg"):
                    file_name = os.path.basename(member.name)
                    try:
                        image_id = int(file_name.split("_")[1].split(".")[0])
                    except (IndexError, ValueError):
                        continue

                    if image_id in ids_set:
                        f = tar.extractfile(member)
                        if f is None:
                            continue

                        image_bytes = f.read()

                        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                        # Convert labels to be 0-indexed
                        label = int(labels_data[image_id - 1])

                        yield (
                            image_id,
                            {
                                "image": image,
                                "label": label,
                            },
                        )

    @staticmethod
    def _labels():
        """Returns the list of 102 flower names corresponding to indices 0-101."""
        return [
            "pink primrose",
            "hard-leaved pocket orchid",
            "canterbury bells",
            "sweet pea",
            "english marigold",
            "tiger lily",
            "moon orchid",
            "bird of paradise",
            "monkshood",
            "globe thistle",
            "snapdragon",
            "colt's foot",
            "king protea",
            "spear thistle",
            "yellow iris",
            "globe-flower",
            "purple coneflower",
            "peruvian lily",
            "balloon flower",
            "giant white arum lily",
            "fire lily",
            "pincushion flower",
            "fritillary",
            "red ginger",
            "grape hyacinth",
            "corn poppy",
            "prince of wales feathers",
            "stemless gentian",
            "artichoke",
            "sweet william",
            "carnation",
            "garden phlox",
            "love in the mist",
            "mexican aster",
            "alpine sea holly",
            "ruby-lipped cattleya",
            "cape flower",
            "great masterwort",
            "siam tulip",
            "lenten rose",
            "barbeton daisy",
            "daffodil",
            "sword lily",
            "poinsettia",
            "bolero deep blue",
            "wallflower",
            "marigold",
            "buttercup",
            "oxeye daisy",
            "common dandelion",
            "petunia",
            "wild pansy",
            "primula",
            "sunflower",
            "pelargonium",
            "bishop of llandaff",
            "gaura",
            "geranium",
            "orange dahlia",
            "pink-yellow dahlia?",
            "cautleya spicata",
            "japanese anemone",
            "black-eyed susan",
            "silverbush",
            "californian poppy",
            "osteospermum",
            "spring crocus",
            "bearded iris",
            "windflower",
            "tree poppy",
            "gazania",
            "azalea",
            "water lily",
            "rose",
            "thorn apple",
            "morning glory",
            "passion flower",
            "lotus",
            "toad lily",
            "anthurium",
            "frangipani",
            "clematis",
            "hibiscus",
            "columbine",
            "desert-rose",
            "tree mallow",
            "magnolia",
            "cyclamen",
            "watercress",
            "canna lily",
            "hippeastrum",
            "bee balm",
            "ball moss",
            "foxglove",
            "bougainvillea",
            "camellia",
            "mallow",
            "mexican petunia",
            "bromelia",
            "blanket flower",
            "trumpet creeper",
            "blackberry lily",
        ]
