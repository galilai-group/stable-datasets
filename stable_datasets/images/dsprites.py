import os

import numpy as np
from PIL import Image as PILImage

from stable_datasets.schema import (
    BuilderConfig,
    DatasetInfo,
    Features,
    Sequence,
    Value,
    Version,
)
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


DSPRITES_VERSION = Version("1.0.0")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_SCREAM_PATH = os.path.join(_PROJECT_ROOT, "docs", "source", "datasets", "imgs", "scream.png")

_DSPRITES_URL = "https://github.com/google-deepmind/dsprites-dataset/raw/refs/heads/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"

_CITATION_HIGGINS = """@inproceedings{higgins2017beta,
                    title={beta-vae: Learning basic visual concepts with a constrained variational framework},
                    author={Higgins, Irina and Matthey, Loic and Pal, Arka and Burgess, Christopher and Glorot, Xavier and Botvinick, Matthew and Mohamed, Shakir and Lerchner, Alexander},
                    booktitle={International conference on learning representations},
                    year={2017}"""

_CITATION_LOCATELLO = """@inproceedings{locatello2019challenging,
                    title={Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations},
                    author={Locatello, Francesco and Bauer, Stefan and Lucic, Mario and Raetsch, Gunnar and Gelly, Sylvain and Sch{\\\"o}lkopf, Bernhard and Bachem, Olivier},
                    booktitle={International Conference on Machine Learning},
                    pages={4114--4124},
                    year={2019}
                    }"""


class DSpritesConfig(BuilderConfig):
    """BuilderConfig for DSprites variants.

    Args:
        has_color_rgb: Whether this variant produces a ``colorRGB`` field.
    """

    def __init__(self, *, has_color_rgb: bool = False, **kwargs):
        super().__init__(version=DSPRITES_VERSION, **kwargs)
        self.has_color_rgb = has_color_rgb


class DSprites(BaseDatasetBuilder):
    """dSprites dataset family.

    dSprites is a dataset of 2D shapes procedurally generated from 6 ground
    truth independent latent factors. These factors are color, shape, scale,
    rotation, x and y positions of a sprite.

    Four variants are available via ``config_name``:

    - ``original``: 64x64 binary grayscale images (default).
    - ``color``: Object rendered with a random RGB color on a black background.
    - ``noise``: White object on a random-noise RGB background.
    - ``scream``: Object rendered by inverting pixels on a Scream painting patch.
    """

    VERSION = DSPRITES_VERSION

    BUILDER_CONFIGS = [
        DSpritesConfig(
            name="original",
            description="Original grayscale dSprites (64x64 binary black-and-white)",
        ),
        DSpritesConfig(
            name="color",
            description="Color variant: random RGB object on black background (64x64x3)",
            has_color_rgb=True,
        ),
        DSpritesConfig(
            name="noise",
            description="Noise variant: white object on random-noise background (64x64x3)",
        ),
        DSpritesConfig(
            name="scream",
            description="Scream variant: object inverted on Scream painting patch (64x64x3)",
        ),
    ]

    def _source(self) -> dict:
        variant = self.config.name
        citation = _CITATION_HIGGINS if variant in ("original", "scream") else _CITATION_LOCATELLO

        return {
            "homepage": "https://github.com/deepmind/dsprites-dataset",
            "assets": {"train": _DSPRITES_URL},
            "citation": citation,
        }

    def _info(self):
        source = self._source()

        features_dict = {
            "image": ImageFeature(),
            "index": Value("int32"),
            "label": Sequence(Value("int32")),
            "label_values": Sequence(Value("float32")),
            "color": Value("int32"),
            "shape": Value("int32"),
            "scale": Value("int32"),
            "orientation": Value("int32"),
            "posX": Value("int32"),
            "posY": Value("int32"),
            "colorValue": Value("float64"),
        }

        if getattr(self.config, "has_color_rgb", False):
            features_dict["colorRGB"] = Sequence(Value("float32"))

        features_dict.update(
            {
                "shapeValue": Value("float64"),
                "scaleValue": Value("float64"),
                "orientationValue": Value("float64"),
                "posXValue": Value("float64"),
                "posYValue": Value("float64"),
            }
        )

        return DatasetInfo(
            description=f"dSprites dataset ({self.config.name} variant).",
            features=Features(features_dict),
            supervised_keys=("image", "label"),
            homepage=source["homepage"],
            citation=source["citation"],
        )

    def _generate_examples(self, data_path, split):
        data = np.load(data_path, allow_pickle=True)
        images = data["imgs"]  # (737280, 64, 64), uint8
        latents_classes = data["latents_classes"]  # (737280, 6), int64
        latents_values = data["latents_values"]  # (737280, 6), float64

        variant = self.config.name

        # Pre-load scream background once
        scream = None
        if variant == "scream":
            scream_img = PILImage.open(_SCREAM_PATH).convert("RGB")
            scream_img = scream_img.resize((350, 274))
            scream = np.array(scream_img).astype(np.float32) / 255.0

        for idx in range(len(images)):
            img = images[idx]  # (64, 64), uint8
            factors_classes = latents_classes[idx].tolist()
            factors_values = latents_values[idx].tolist()

            color_rgb = None

            if variant == "original":
                img_pil = PILImage.fromarray(img * 255, mode="L")

            elif variant == "color":
                img_f = img.astype(np.float32)
                color_rgb = np.random.uniform(0.5, 1.0, size=(3,))
                img_rgb = (img_f[..., None] * color_rgb) * 255
                img_pil = PILImage.fromarray(img_rgb.astype(np.uint8), mode="RGB")

            elif variant == "noise":
                img_f = img.astype(np.float32)
                noise = np.random.uniform(0, 1, size=(64, 64, 3))
                img_rgb = np.minimum(img_f[..., None] + noise, 1.0) * 255
                img_pil = PILImage.fromarray(img_rgb.astype(np.uint8), mode="RGB")

            elif variant == "scream":
                img_f = img.astype(np.float32)
                x_crop = np.random.randint(0, scream.shape[0] - 64)
                y_crop = np.random.randint(0, scream.shape[1] - 64)
                background_patch = scream[x_crop : x_crop + 64, y_crop : y_crop + 64]
                mask = img_f == 1
                output_img = np.copy(background_patch)
                output_img[mask] = 1.0 - background_patch[mask]
                img_pil = PILImage.fromarray((output_img * 255).astype(np.uint8), mode="RGB")

            example = {
                "image": img_pil,
                "index": idx,
                "label": factors_classes,
                "label_values": factors_values,
                "color": factors_classes[0],
                "shape": factors_classes[1],
                "scale": factors_classes[2],
                "orientation": factors_classes[3],
                "posX": factors_classes[4],
                "posY": factors_classes[5],
                "colorValue": factors_values[0],
                "shapeValue": factors_values[1],
                "scaleValue": factors_values[2],
                "orientationValue": factors_values[3],
                "posXValue": factors_values[4],
                "posYValue": factors_values[5],
            }

            if variant == "color":
                example["colorRGB"] = color_rgb.tolist()

            yield idx, example
