import datasets
import scipy.io as sio
from PIL import Image

from stable_datasets.utils import BaseDatasetBuilder


class SVHN(BaseDatasetBuilder):
    """SVHN (Street View House Numbers) Dataset for image classification.

    SVHN is a real-world image dataset for developing machine learning and object recognition algorithms
    with minimal requirement on data preprocessing and formatting. It can be seen as similar in flavor to MNIST,
    but incorporates an order of magnitude more labeled data (over 600,000 digit images) and comes from a significantly harder,
    unsolved, real world problem (recognizing digits and numbers in natural scene images). SVHN is obtained from house numbers
    in Google Street View images.
    """

    VERSION = datasets.Version("1.0.0")

    SOURCE = {
        "homepage": "http://ufldl.stanford.edu/housenumbers/",
        "assets": {
            "train": "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
            "test": "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
            "extra": "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
        },
        "citation": """@inproceedings{netzer2011reading,
                          title={Reading digits in natural images with unsupervised feature learning},
                          author={Netzer, Yuval and Wang, Tao and Coates, Adam and Bissacco, Alessandro and Wu, Baolin and Ng, Andrew Y and others},
                          booktitle={NIPS workshop on deep learning and unsupervised feature learning},
                          volume={2011},
                          number={2},
                          pages={4},
                          year={2011},
                          organization={Granada}
                        }""",
    }

    def _info(self):
        return datasets.DatasetInfo(
            description="""The Street View House Numbers (SVHN) Dataset is a real-world image dataset
                           for developing machine learning and object recognition algorithms with minimal
                           requirement on data preprocessing and formatting. SVHN is obtained from house
                           numbers in Google Street View images.""",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(names=[str(i) for i in range(10)]),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        """Generate examples from the MAT file."""
        mat = sio.loadmat(str(data_path), squeeze_me=True)

        X = mat["X"]  # Shape: (32, 32, 3, N)
        y = mat["y"]  # Shape: (N,)

        # Transpose to get (N, 32, 32, 3)
        X = X.transpose(3, 0, 1, 2)

        # In SVHN, label 10 represents digit 0
        y[y == 10] = 0

        for idx in range(len(y)):
            yield (
                idx,
                {
                    "image": Image.fromarray(X[idx]),
                    "label": int(y[idx]),
                },
            )
