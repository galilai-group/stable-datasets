import zipfile

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


# Pinned to a commit so the split files and images can't change under us.
_ARCHIVE = DownloadInfo(
    url="https://github.com/marcusklasson/GroceryStoreDataset/archive/fc80ba90f803d79d0383df52c5a4ac5de99ff6fc.zip",
    filename="GroceryStoreDataset.zip",
)

# Fine-grained classes, indexed by "Class ID" in dataset/classes.csv.
_FINE_CLASSES = [
    "Golden-Delicious",
    "Granny-Smith",
    "Pink-Lady",
    "Red-Delicious",
    "Royal-Gala",
    "Avocado",
    "Banana",
    "Kiwi",
    "Lemon",
    "Lime",
    "Mango",
    "Cantaloupe",
    "Galia-Melon",
    "Honeydew-Melon",
    "Watermelon",
    "Nectarine",
    "Orange",
    "Papaya",
    "Passion-Fruit",
    "Peach",
    "Anjou",
    "Conference",
    "Kaiser",
    "Pineapple",
    "Plum",
    "Pomegranate",
    "Red-Grapefruit",
    "Satsumas",
    "Bravo-Apple-Juice",
    "Bravo-Orange-Juice",
    "God-Morgon-Apple-Juice",
    "God-Morgon-Orange-Juice",
    "God-Morgon-Orange-Red-Grapefruit-Juice",
    "God-Morgon-Red-Grapefruit-Juice",
    "Tropicana-Apple-Juice",
    "Tropicana-Golden-Grapefruit",
    "Tropicana-Juice-Smooth",
    "Tropicana-Mandarin-Morning",
    "Arla-Ecological-Medium-Fat-Milk",
    "Arla-Lactose-Medium-Fat-Milk",
    "Arla-Medium-Fat-Milk",
    "Arla-Standard-Milk",
    "Garant-Ecological-Medium-Fat-Milk",
    "Garant-Ecological-Standard-Milk",
    "Oatly-Natural-Oatghurt",
    "Oatly-Oat-Milk",
    "Arla-Ecological-Sour-Cream",
    "Arla-Sour-Cream",
    "Arla-Sour-Milk",
    "Alpro-Blueberry-Soyghurt",
    "Alpro-Vanilla-Soyghurt",
    "Alpro-Fresh-Soy-Milk",
    "Alpro-Shelf-Soy-Milk",
    "Arla-Mild-Vanilla-Yoghurt",
    "Arla-Natural-Mild-Low-Fat-Yoghurt",
    "Arla-Natural-Yoghurt",
    "Valio-Vanilla-Yoghurt",
    "Yoggi-Strawberry-Yoghurt",
    "Yoggi-Vanilla-Yoghurt",
    "Asparagus",
    "Aubergine",
    "Cabbage",
    "Carrots",
    "Cucumber",
    "Garlic",
    "Ginger",
    "Leek",
    "Brown-Cap-Mushroom",
    "Yellow-Onion",
    "Green-Bell-Pepper",
    "Orange-Bell-Pepper",
    "Red-Bell-Pepper",
    "Yellow-Bell-Pepper",
    "Floury-Potato",
    "Solid-Potato",
    "Sweet-Potato",
    "Red-Beet",
    "Beef-Tomato",
    "Regular-Tomato",
    "Vine-Tomato",
    "Zucchini",
]

# Coarse-grained classes, indexed by "Coarse Class ID" in dataset/classes.csv.
_COARSE_CLASSES = [
    "Apple",
    "Avocado",
    "Banana",
    "Kiwi",
    "Lemon",
    "Lime",
    "Mango",
    "Melon",
    "Nectarine",
    "Orange",
    "Papaya",
    "Passion-Fruit",
    "Peach",
    "Pear",
    "Pineapple",
    "Plum",
    "Pomegranate",
    "Red-Grapefruit",
    "Satsumas",
    "Juice",
    "Milk",
    "Oatghurt",
    "Oat-Milk",
    "Sour-Cream",
    "Sour-Milk",
    "Soyghurt",
    "Soy-Milk",
    "Yoghurt",
    "Asparagus",
    "Aubergine",
    "Cabbage",
    "Carrots",
    "Cucumber",
    "Garlic",
    "Ginger",
    "Leek",
    "Mushroom",
    "Onion",
    "Pepper",
    "Potato",
    "Red-Beet",
    "Tomato",
    "Zucchini",
]


class GroceryStore(BaseDatasetBuilder):
    """Grocery Store dataset: natural images of fruits, vegetables and packaged goods.

    Smartphone photos taken in grocery stores, labeled with 81 fine-grained classes
    (e.g. "Granny-Smith") that group into 43 coarse-grained classes (e.g. "Apple").
    The official split has 2,640 train, 296 validation and 2,485 test images. Note that
    the validation set only covers 60 of the 81 fine-grained classes.

    Each example has the image, the fine-grained ``label`` and the ``coarse_label``.
    """

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://github.com/marcusklasson/GroceryStoreDataset",
        assets={
            "train": _ARCHIVE,
            "validation": _ARCHIVE,
            "test": _ARCHIVE,
        },
        citation=r"""@inproceedings{klasson2019hierarchical,
                         title={A Hierarchical Grocery Store Image Dataset with Visual and Semantic Labels},
                         author={Klasson, Marcus and Zhang, Cheng and Kjellstr{\"o}m, Hedvig},
                         booktitle={IEEE Winter Conference on Applications of Computer Vision (WACV)},
                         year={2019}}""",
        license="MIT License",
    )

    def _info(self):
        return DatasetInfo(
            description="""Grocery Store dataset of 5,421 natural images of grocery items (fruits, vegetables,
                and carton items such as juice, milk and yoghurt) taken with a smartphone camera in grocery
                stores. Images have 81 fine-grained labels and 43 coarse-grained labels.""",
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=_FINE_CLASSES),
                    "coarse_label": ClassLabel(names=_COARSE_CLASSES),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license=self.SOURCE["license"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        list_name = "val" if split == "validation" else split
        with zipfile.ZipFile(data_path, "r") as archive:
            # GitHub archives wrap everything in a "<repo>-<commit>/" folder.
            root = next(n for n in archive.namelist() if n.endswith("dataset/classes.csv"))[: -len("classes.csv")]
            lines = archive.read(f"{root}{list_name}.txt").decode("utf-8").splitlines()
            for line in lines:
                if not line.strip():
                    continue
                # Each line is "<relative image path>, <fine label>, <coarse label>".
                path, fine, coarse = (part.strip() for part in line.split(","))
                yield (
                    path,
                    {
                        "image": archive.read(root + path),
                        "label": int(fine),
                        "coarse_label": int(coarse),
                    },
                )
