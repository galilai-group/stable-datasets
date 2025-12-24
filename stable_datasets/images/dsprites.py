import datasets
import numpy as np
from PIL import Image

from stable_datasets.utils import BaseDatasetBuilder


class DSprites(BaseDatasetBuilder):
    """DSprites
    dSprites is a dataset of 2D shapes procedurally generated from 6 ground truth independent latent factors. These factors are color, shape, scale, rotation, x and y positions of a sprite."""

    VERSION = datasets.Version("1.0.0")

    SOURCE = {
        "homepage": "https://github.com/deepmind/dsprites-dataset",
        "assets": {
            "train": "https://github.com/google-deepmind/dsprites-dataset/raw/refs/heads/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
        },
        "citation": """@inproceedings{higgins2017beta,
                    title={beta-vae: Learning basic visual concepts with a constrained variational framework},
                    author={Higgins, Irina and Matthey, Loic and Pal, Arka and Burgess, Christopher and Glorot, Xavier and Botvinick, Matthew and Mohamed, Shakir and Lerchner, Alexander},
                    booktitle={International conference on learning representations},
                    year={2017}""",
    }

    def _info(self):
        return datasets.DatasetInfo(
            description=""""dSprites dataset: procedurally generated 2D shapes dataset with known ground-truth factors, "
                "commonly used for disentangled representation learning. "
                "Factors: color (1), shape (3), scale (6), orientation (40), position X (32), position Y (32). "
                "Images are 64x64 binary black-and-white.""",
            features=datasets.Features(
                {
                    "image": datasets.Image(),  # (64, 64), grayscale
                    "index": datasets.Value("int32"),  # index of the image
                    "label": datasets.Sequence(datasets.Value("int32")),  # 6 factor indices (classes)
                    "label_values": datasets.Sequence(datasets.Value("float32")),  # 6 factor continuous values
                    "color": datasets.Value("int32"),  # color index (always 0)
                    "shape": datasets.Value("int32"),  # shape index (0-2)
                    "scale": datasets.Value("int32"),  # scale index (0-5)
                    "orientation": datasets.Value("int32"),  # orientation index (0-39)
                    "posX": datasets.Value("int32"),  # posX index (0-31)
                    "posY": datasets.Value("int32"),  # posY index (0-31)
                    "colorValue": datasets.Value("float64"),  # color value (always 1.0)
                    "shapeValue": datasets.Value("float64"),  # shape value (0, 1, 2)
                    "scaleValue": datasets.Value("float64"),  # scale value (0.5, 1)
                    "orientationValue": datasets.Value("float64"),  # orientation value (0, 2pi)
                    "posXValue": datasets.Value("float64"),  # posX value (0, 1)
                    "posYValue": datasets.Value("float64"),  # posY value (0, 1)
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        # Load npz
        data = np.load(data_path, allow_pickle=True)
        images = data["imgs"]  # shape: (737280, 64, 64), uint8
        latents_classes = data["latents_classes"]  # shape: (737280, 6), int64
        latents_values = data["latents_values"]  # shape: (737280, 6), float64

        # Iterate over images
        for idx in range(len(images)):
            img = images[idx]  # (64, 64), uint8
            img = img * 255
            # Convert to PIL image, keep grayscale mode
            img_pil = Image.fromarray(img, mode="L")

            factors_classes = latents_classes[
                idx
            ].tolist()  # [color_idx, shape_idx, scale_idx, orientation_idx, posX_idx, posY_idx]
            factors_values = latents_values[idx].tolist()

            yield (
                idx,
                {
                    "image": img_pil,
                    "index": idx,
                    "label": factors_classes,
                    "label_values": factors_values,
                    "color": factors_classes[0],  # always 0
                    "shape": factors_classes[1],
                    "scale": factors_classes[2],
                    "orientation": factors_classes[3],
                    "posX": factors_classes[4],
                    "posY": factors_classes[5],
                    "colorValue": factors_values[0],  # always 0.0
                    "shapeValue": factors_values[1],
                    "scaleValue": factors_values[2],
                    "orientationValue": factors_values[3],
                    "posXValue": factors_values[4],
                    "posYValue": factors_values[5],
                },
            )
