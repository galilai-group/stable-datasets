import datasets
import numpy as np
from PIL import Image

from stable_datasets.utils import BaseDatasetBuilder


class Shapes3D(BaseDatasetBuilder):
    """Shapes3D dataset: 10x10x10x8x4x15 factor combinations, 64x64 RGB images."""

    VERSION = datasets.Version("1.0.0")

    SOURCE = {
        "homepage": "https://github.com/google-deepmind/3dshapes-dataset/",
        "assets": {
            "train": "https://huggingface.co/datasets/randall-lab/shapes3d/resolve/main/shapes3d.npz",
        },
        "license": "apache-2.0",
        "citation": """@InProceedings{pmlr-v80-kim18b,
  title = {Disentangling by Factorising},
  author = {Kim, Hyunjik and Mnih, Andriy},
  booktitle = {Proceedings of the 35th International Conference on Machine Learning},
  pages = {2649--2658},
  year = {2018},
  editor = {Dy, Jennifer and Krause, Andreas},
  volume = {80},
  series = {Proceedings of Machine Learning Research},
  month = {10--15 Jul},
  publisher = {PMLR},
  pdf = {http://proceedings.mlr.press/v80/kim18b/kim18b.pdf},
  url = {https://proceedings.mlr.press/v80/kim18b.html}
}""",
    }

    def _info(self):
        return datasets.DatasetInfo(
            description=(
                "Shapes3D dataset: procedurally generated images of 3D shapes with 6 independent factors of variation. "
                "Commonly used for disentangled representation learning. "
                "Factors: floor hue (10), wall hue (10), object hue (10), scale (8), shape (4), orientation (15). "
                "Images are stored as the Cartesian product of the factors in row-major order."
            ),
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.Sequence(datasets.Value("float64")),
                    "label_index": datasets.Sequence(datasets.Value("int64")),
                    "floor": datasets.Value("float64"),
                    "wall": datasets.Value("float64"),
                    "object": datasets.Value("float64"),
                    "scale": datasets.Value("float64"),
                    "shape": datasets.Value("float64"),
                    "orientation": datasets.Value("float64"),
                    "floor_idx": datasets.Value("int32"),
                    "wall_idx": datasets.Value("int32"),
                    "object_idx": datasets.Value("int32"),
                    "scale_idx": datasets.Value("int32"),
                    "shape_idx": datasets.Value("int32"),
                    "orientation_idx": datasets.Value("int32"),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license=self.SOURCE["license"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        data = np.load(data_path)
        images = data["images"]
        labels = data["labels"]

        factor_sizes = np.array([10, 10, 10, 8, 4, 15])
        factor_bases = np.cumprod([1] + list(factor_sizes[::-1]))[::-1][1:]

        def index_to_factors(index: int) -> list[int]:
            return [int((index // int(base)) % int(size)) for base, size in zip(factor_bases, factor_sizes)]

        for idx in range(len(images)):
            img_pil = Image.fromarray(images[idx])

            label_value = labels[idx].tolist()
            label_index = index_to_factors(idx)

            yield (
                idx,
                {
                    "image": img_pil,
                    "label": label_value,
                    "label_index": label_index,
                    "floor": label_value[0],
                    "wall": label_value[1],
                    "object": label_value[2],
                    "scale": label_value[3],
                    "shape": label_value[4],
                    "orientation": label_value[5],
                    "floor_idx": label_index[0],
                    "wall_idx": label_index[1],
                    "object_idx": label_index[2],
                    "scale_idx": label_index[3],
                    "shape_idx": label_index[4],
                    "orientation_idx": label_index[5],
                },
            )
