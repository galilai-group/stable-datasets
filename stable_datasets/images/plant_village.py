import io
from zipfile import ZipFile

from PIL import Image as PILImage
from tqdm import tqdm

from stable_datasets.schema import (
    BuilderConfig,
    ClassLabel,
    DatasetInfo,
    Features,
    Value,
    Version,
)
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, bulk_download


PLANT_VILLAGE_VERSION = Version("1.0.0")

_VARIANTS = ("color", "grayscale", "segmented")

_HF_REPO = "https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main"

_CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

_CLASS_INDEX = {name: i for i, name in enumerate(_CLASS_NAMES)}


class PlantVillageConfig(BuilderConfig):
    """One BuilderConfig per image variant (color / grayscale / segmented)."""

    def __init__(self, *, variant: str, **kwargs):
        super().__init__(version=PLANT_VILLAGE_VERSION, **kwargs)
        if variant not in _VARIANTS:
            raise ValueError(f"variant={variant!r} is not valid; choose from {_VARIANTS}.")
        self.variant = variant


class PlantVillage(BaseDatasetBuilder):
    """PlantVillage Dataset

    The PlantVillage dataset is an open-access repository of leaf images covering 14 crop species
    and 26 plant diseases (plus healthy controls), for 38 classes in total and ~54,000 images.
    Three image variants are provided:

    - ``color``: original RGB photographs of the leaves
    - ``grayscale``: the same images converted to grayscale
    - ``segmented``: leaves segmented from the background

    The train/test split used here is the one published alongside the HuggingFace mirror
    (``mohanty/PlantVillage``), with ~80/20 partitioning per class.
    """

    VERSION = PLANT_VILLAGE_VERSION

    BUILDER_CONFIGS = [
        PlantVillageConfig(
            name="color",
            description="Original RGB leaf photographs.",
            variant="color",
        ),
        PlantVillageConfig(
            name="grayscale",
            description="Grayscale leaf photographs.",
            variant="grayscale",
        ),
        PlantVillageConfig(
            name="segmented",
            description="Leaves segmented from the background.",
            variant="segmented",
        ),
    ]
    DEFAULT_CONFIG_NAME = "color"

    def _source(self):
        variant = self.config.variant
        return {
            "homepage": "https://github.com/spMohanty/PlantVillage-Dataset",
            "assets": {
                "data": f"{_HF_REPO}/data.zip",
                "train_split": f"{_HF_REPO}/splits/{variant}_train.txt",
                "test_split": f"{_HF_REPO}/splits/{variant}_test.txt",
            },
            "citation": """@article{hughes2015open,
                title={An open access repository of images on plant health to enable the development of mobile disease diagnostics},
                author={Hughes, David P. and Salath{\\'e}, Marcel},
                journal={arXiv preprint arXiv:1511.08060},
                year={2015}
            }
            @article{mohanty2016using,
                title={Using deep learning for image-based plant disease detection},
                author={Mohanty, Sharada P. and Hughes, David P. and Salath{\\'e}, Marcel},
                journal={Frontiers in Plant Science},
                volume={7},
                pages={1419},
                year={2016}
            }""",
        }

    def _info(self):
        source = self._source()
        return DatasetInfo(
            description=(
                "PlantVillage plant-disease image dataset covering 14 crops and 26 diseases "
                f"(38 classes total) in three image variants: {', '.join(_VARIANTS)}."
            ),
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=_CLASS_NAMES),
                    "crop": Value("string"),
                    "disease": Value("string"),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=source["homepage"],
            citation=source["citation"],
        )

    def _split_generators(self):
        source = self._source()
        key_url_map = {
            "data": source["assets"]["data"],
            "train_split": source["assets"]["train_split"],
            "test_split": source["assets"]["test_split"],
        }
        urls = list(key_url_map.values())
        local_paths = bulk_download(urls, dest_folder=self._raw_download_dir)
        path_map = dict(zip(key_url_map.keys(), local_paths))

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"path_map": path_map, "split": "train"},
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={"path_map": path_map, "split": "test"},
            ),
        ]

    def _generate_examples(self, path_map, split):
        """For each entry in the split file, read its image from data.zip and derive crop/disease from the path."""
        data_path = path_map["data"]
        split_path = path_map["train_split"] if split == "train" else path_map["test_split"]

        with open(split_path) as f:
            rel_paths = [line.strip() for line in f if line.strip()]

        with ZipFile(data_path, "r") as archive:
            for rel_path in tqdm(rel_paths, desc=f"Processing {self.config.variant} {split} set"):
                parts = rel_path.split("/")
                # Expect: raw/<variant>/<Crop___Disease>/<file>
                if len(parts) < 4:
                    continue
                class_name = parts[2]
                sub = class_name.split("___", 1)
                crop = sub[0]
                disease = sub[1] if len(sub) > 1 else "unknown"

                try:
                    raw_bytes = archive.read(rel_path)
                except KeyError:
                    continue
                image = PILImage.open(io.BytesIO(raw_bytes)).convert("RGB")

                yield (
                    rel_path,
                    {
                        "image": image,
                        "label": _CLASS_INDEX[class_name],
                        "crop": crop,
                        "disease": disease,
                    },
                )
