import os
import tarfile

import datasets
import numpy as np
import scipy.io
from PIL import Image

from stable_datasets.utils import BaseDatasetBuilder


class CARS3D(BaseDatasetBuilder):
    """CARS3D
    183 car types x 24 azimuth angles x 4 elevation angles.
    """

    VERSION = datasets.Version("1.0.0")

    SOURCE = {
        "homepage": "https://github.com/google-research/disentanglement_lib/tree/master",
        "assets": {
            "train": "http://www.scottreed.info/files/nips2015-analogy-data.tar.gz",
        },
        "license": "Apache-2.0",
        "citation": """@inproceedings{locatello2019challenging,
  title={Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations},
  author={Locatello, Francesco and Bauer, Stefan and Lucic, Mario and Raetsch, Gunnar and Gelly, Sylvain and Sch{\"o}lkopf, Bernhard and Bachem, Olivier},
  booktitle={International Conference on Machine Learning},
  pages={4114--4124},
  year={2019}
}""",
    }

    def _info(self):
        return datasets.DatasetInfo(
            description=(
                "Cars3D dataset with 183 car types, 24 azimuth angles, 4 elevation angles. Images are 128x128 RGB."
            ),
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "car_type": datasets.Value("int32"),
                    "elevation": datasets.Value("int32"),
                    "azimuth": datasets.Value("int32"),
                    "label": datasets.Sequence(datasets.Value("int32")),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license=self.SOURCE["license"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        with tarfile.open(data_path, "r:gz") as tar:
            tar.extractall(path=os.path.dirname(data_path))

        mat_dir = os.path.join(os.path.dirname(data_path), "data", "cars")

        mat_files = sorted(f for f in os.listdir(mat_dir) if f.endswith(".mat"))

        idx = 0
        for car_idx, mat_file in enumerate(mat_files):
            mat_path = os.path.join(mat_dir, mat_file)
            mat_data = scipy.io.loadmat(mat_path)
            im_data = mat_data["im"]

            for elev in range(im_data.shape[4]):
                for azim in range(im_data.shape[3]):
                    img = im_data[:, :, :, azim, elev]
                    img = img.astype(np.uint8)
                    img_pil = Image.fromarray(img, mode="RGB")

                    yield (
                        idx,
                        {
                            "image": img_pil,
                            "car_type": car_idx,
                            "elevation": elev,
                            "azimuth": azim,
                            "label": [car_idx, elev, azim],
                        },
                    )
                    idx += 1
