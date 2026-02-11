import tarfile

import datasets
import numpy as np

from stable_datasets.utils import BaseDatasetBuilder


class STL10(BaseDatasetBuilder):
    """STL-10 Dataset"""

    VERSION = datasets.Version("1.0.0")

    # Single source-of-truth for dataset provenance + download locations.
    SOURCE = {
        "homepage": "https://cs.stanford.edu/~acoates/stl10/",
        "assets": {
            "train": "https://cs.stanford.edu/~acoates/stl10/stl10_binary.tar.gz",
            "test": "https://cs.stanford.edu/~acoates/stl10/stl10_binary.tar.gz",
            "unlabeled": "https://cs.stanford.edu/~acoates/stl10/stl10_binary.tar.gz",
        },
        "citation": """@article{coates2011analysis,
                        title={An analysis of single-layer networks in unsupervised feature learning},
                        author={Coates, Adam and Ng, Andrew Y},
                        journal={AISTATS},
                        year={2011}}""",
    }

    def _info(self):
        return datasets.DatasetInfo(
            description="STL-10 dataset for unsupervised feature learning. Includes labeled and unlabeled images.",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(
                        names=["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
                    ),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        """Generate examples from the tar.gz archive."""
        with tarfile.open(data_path, "r:gz") as tar:
            if split == "train":
                images_file = "stl10_binary/train_X.bin"
                labels_file = "stl10_binary/train_y.bin"
            elif split == "test":
                images_file = "stl10_binary/test_X.bin"
                labels_file = "stl10_binary/test_y.bin"
            else:  # unlabeled
                images_file = "stl10_binary/unlabeled_X.bin"
                labels_file = None

            images = tar.extractfile(images_file).read()
            # STL-10 binary is stored in column-major (Fortran) order from MATLAB
            # After reshape we have (N, C, 96, 96), transpose (0,3,2,1) to get correct (N, H, W, C)
            images = np.frombuffer(images, dtype=np.uint8).reshape(-1, 3, 96, 96).transpose((0, 3, 2, 1))

            if labels_file:
                labels = tar.extractfile(labels_file).read()
                labels = np.frombuffer(labels, dtype=np.uint8) - 1
                for idx, (image, label) in enumerate(zip(images, labels)):
                    yield idx, {"image": image, "label": int(label)}
            else:
                for idx, image in enumerate(images):
                    yield idx, {"image": image, "label": -1}
