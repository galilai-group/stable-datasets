from zipfile import ZipFile

import datasets
import numpy as np
from PIL import Image

from stable_datasets.utils import BaseDatasetBuilder


def _read_binary_matrix_from_bytes(b: bytes):
    magic = int(np.frombuffer(b, dtype=np.int32, count=1)[0])
    ndim = int(np.frombuffer(b, dtype=np.int32, count=1, offset=4)[0])
    eff_dim = max(3, ndim)
    raw_dims = np.frombuffer(b, "int32", eff_dim, 8)

    dims = [int(raw_dims[i]) for i in range(ndim)]

    dtype_map = {
        507333717: "int8",
        507333716: "int32",
        507333713: "float",
        507333715: "double",
    }
    dtype = dtype_map[magic]

    data = np.frombuffer(b, dtype, offset=8 + eff_dim * 4)
    return data.reshape(tuple(dims))


def _load_small_norb_from_zip(zip_path: str):
    with ZipFile(zip_path, "r") as zf:
        dat_name = next(n for n in zf.namelist() if n.endswith("-dat.mat"))
        cat_name = next(n for n in zf.namelist() if n.endswith("-cat.mat"))
        info_name = next(n for n in zf.namelist() if n.endswith("-info.mat"))

        dat_bytes = zf.read(dat_name)
        cat_bytes = zf.read(cat_name)
        info_bytes = zf.read(info_name)

    norb = _read_binary_matrix_from_bytes(dat_bytes)
    images_left = norb[:, 0]
    images_right = norb[:, 1]

    norb_class = _read_binary_matrix_from_bytes(cat_bytes)
    norb_info = _read_binary_matrix_from_bytes(info_bytes)

    features = np.column_stack((norb_class, norb_info)).astype(np.int32)
    features[:, 3] = (features[:, 3] // 2).astype(np.int32)

    return images_left, images_right, features


class SmallNORB(BaseDatasetBuilder):
    """SmallNORB dataset: 96x96 stereo images with 5 known factors."""

    VERSION = datasets.Version("1.0.0")

    SOURCE = {
        "homepage": "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/",
        "assets": {
            "train": "https://huggingface.co/datasets/randall-lab/small-norb/resolve/main/smallnorb-train.zip",
            "test": "https://huggingface.co/datasets/randall-lab/small-norb/resolve/main/smallnorb-test.zip",
        },
        "license": "Apache-2.0",
        "citation": """@inproceedings{lecun2004learning,
  title={Learning methods for generic object recognition with invariance to pose and lighting},
  author={LeCun, Yann and Huang, Fu Jie and Bottou, Leon},
  booktitle={Proceedings of the 2004 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2004. CVPR 2004.},
  volume={2},
  pages={II--104},
  year={2004},
  organization={IEEE}
}""",
    }

    def _info(self):
        return datasets.DatasetInfo(
            description=(
                "SmallNORB dataset: stereo pair images of 3D toy objects, used for learning object recognition "
                "robust to pose and lighting. Each image pair corresponds to a combination of 5 factors: "
                "category, instance, elevation, azimuth, lighting."
            ),
            features=datasets.Features(
                {
                    "left_image": datasets.Image(),
                    "right_image": datasets.Image(),
                    "label": datasets.Sequence(datasets.Value("int32")),
                    "category": datasets.Value("int32"),
                    "instance": datasets.Value("int32"),
                    "elevation": datasets.Value("int32"),
                    "azimuth": datasets.Value("int32"),
                    "lighting": datasets.Value("int32"),
                }
            ),
            supervised_keys=("left_image", "label"),
            homepage=self.SOURCE["homepage"],
            license=self.SOURCE["license"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        images_left, images_right, features = _load_small_norb_from_zip(str(data_path))

        for idx in range(len(images_left)):
            left_img = Image.fromarray(images_left[idx].astype(np.uint8), mode="L")
            right_img = Image.fromarray(images_right[idx].astype(np.uint8), mode="L")

            factors = features[idx].tolist()

            yield (
                idx,
                {
                    "left_image": left_img,
                    "right_image": right_img,
                    "label": factors,
                    "category": factors[0],
                    "instance": factors[1],
                    "elevation": factors[2],
                    "azimuth": factors[3],
                    "lighting": factors[4],
                },
            )
