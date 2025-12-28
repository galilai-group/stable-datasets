import io
import zipfile

import datasets
import numpy as np
import scipy.io as sio

from stable_datasets.utils import BaseDatasetBuilder


class EMNISTConfig(datasets.BuilderConfig):
    def __init__(self, variant, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)
        self.variant = variant


class EMNIST(BaseDatasetBuilder):
    """EMNIST (Extended MNIST) Dataset

    Abstract
    EMNIST is a set of handwritten characters derived from the NIST Special Database 19 and converted to a 28x28 pixel format that directly matches the MNIST dataset. It serves as a challenging "drop-in" replacement for MNIST, introducing handwritten letters and a larger variety of writing styles while preserving the original file structure and pixel density.

    Context
    While the original MNIST dataset is considered "solved" by modern architectures, EMNIST restores the challenge by providing a larger, more diverse benchmark. It bridges the gap between simple digit recognition and complex handwriting tasks, offering up to 62 classes (digits + uppercase + lowercase) to test generalization and writer-independent recognition.

    Content
    The dataset contains up to 814,255 grayscale images (28x28). It is provided in six split configurations to suit different needs:
    * **ByClass** & **ByMerge**: Full unbalanced sets (up to 62 classes).
    * **Balanced**: 131,600 images across 47 classes (ideal for benchmarking).
    * **Letters**: 145,600 images across 26 classes (A-Z).
    * **Digits** & **MNIST**: 280,000+ images across 10 classes (0-9).
    """

    VERSION = datasets.Version("1.0.0")

    # Single source-of-truth for dataset provenance + download locations.
    SOURCE = {
        "homepage": "https://www.nist.gov/itl/iad/image-group/emnist-dataset",
        "citation": """@misc{cohen2017emnistextensionmnisthandwritten,
                        title={EMNIST: an extension of MNIST to handwritten letters},
                        author={Gregory Cohen and Saeed Afshar and Jonathan Tapson and André van Schaik},
                        year={2017},
                        eprint={1702.05373},
                        archivePrefix={arXiv},
                        primaryClass={cs.CV},
                        url={https://arxiv.org/abs/1702.05373},
            }""",
        "assets": {
            "train": "https://biometrics.nist.gov/cs_links/EMNIST/matlab.zip",
            "test": "https://biometrics.nist.gov/cs_links/EMNIST/matlab.zip",
        },
    }

    BUILDER_CONFIGS = [
        EMNISTConfig(name="byclass", variant="byclass"),
        EMNISTConfig(name="bymerge", variant="bymerge"),
        EMNISTConfig(name="balanced", variant="balanced"),
        EMNISTConfig(name="letters", variant="letters"),
        EMNISTConfig(name="digits", variant="digits"),
        EMNISTConfig(name="mnist", variant="mnist"),
    ]

    def _info(self):
        variant = self.config.variant
        if variant == "byclass":
            num_classes = 62
        elif variant == "bymerge":
            num_classes = 47
        elif variant == "balanced":
            num_classes = 47
        elif variant == "letters":
            num_classes = 26
        elif variant == "digits":
            num_classes = 10
        elif variant == "mnist":
            num_classes = 10
        else:
            num_classes = 0

        return datasets.DatasetInfo(
            description="""The EMNIST dataset is an image classification dataset of 28x28 grayscale handwritten character images, organized into 6 distinct configurations (ByClass, ByMerge, Balanced, Letters, Digits, MNIST) ranging from 10 to 62 classes. See https://www.nist.gov/itl/iad/image-group/emnist-dataset for more information.""",
            features=datasets.Features(
                {"image": datasets.Image(), "label": datasets.ClassLabel(num_classes=num_classes)}
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        """Generate examples from the ZIP archives of images and labels."""
        variant = self.config.variant
        mat_filename = f"matlab/emnist-{variant}.mat"

        with zipfile.ZipFile(data_path, "r") as z:
            with z.open(mat_filename) as f:
                mat_data_bytes = io.BytesIO(f.read())
                data = sio.loadmat(mat_data_bytes)

        dataset = data["dataset"][0, 0]
        subset = dataset[split][0, 0]

        images = subset["images"]
        labels = subset["labels"]

        images = np.array(images, dtype=np.uint8).reshape(-1, 28, 28)

        labels = np.array(labels, dtype=np.int64).flatten()

        for idx, (img, lbl) in enumerate(zip(images, labels)):
            lbl = int(lbl)
            if variant == "letters":
                lbl -= 1  # Letters are 1-26 in the dataset, shift to 0-25
            yield idx, {"image": img, "label": int(lbl)}
