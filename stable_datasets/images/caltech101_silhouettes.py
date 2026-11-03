import numpy as np
import scipy.io as sio

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


_SILHOUETTES_URL = "https://people.cs.umass.edu/~marlin/data/caltech101_silhouettes_28_split1.mat"


class Caltech101Silhouettes(BaseDatasetBuilder):
    """CalTech 101 Silhouettes (28x28) with the official train/validation/test split.

    Each image is the high-quality polygon outline of the primary object in a
    Caltech-101 image, centered, scaled and rendered as a filled black polygon on
    a white background. The split is stratified: each class contributes
    min(3/5 * N_c, 100) training images, and the remainder is divided evenly
    between validation and test, so the evaluation splits are class-imbalanced.
    """

    VERSION = Version("1.0.0")

    # The .mat file holds all three splits; it is downloaded once.
    SOURCE = DatasetSource(
        homepage="https://people.cs.umass.edu/~marlin/data.shtml",
        assets={
            "train": DownloadInfo(url=_SILHOUETTES_URL),
            "val": DownloadInfo(url=_SILHOUETTES_URL),
            "test": DownloadInfo(url=_SILHOUETTES_URL),
        },
        citation="""@inproceedings{marlin2010inductive,
                        title={Inductive Principles for Restricted Boltzmann Machine Learning},
                        author={Marlin, Benjamin and Swersky, Kevin and Chen, Bo and de Freitas, Nando},
                        booktitle={Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics},
                        pages={509--516},
                        year={2010},
                        volume={9},
                        series={Proceedings of Machine Learning Research},
                        publisher={PMLR}}""",
        license="Unknown",
    )

    def _info(self):
        return DatasetInfo(
            description="""CalTech 101 Silhouettes: 8,671 binary 28x28 silhouettes of the primary object outline in
                Caltech-101 images, across 101 classes, with the authors' official stratified
                train/validation/test split (4,100 / 2,264 / 2,307). The validation and test splits are
                class-imbalanced, so class-balanced accuracy is recommended.""",
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=self._labels()),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license=self.SOURCE["license"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        mat = sio.loadmat(data_path)
        # Pixels are flattened column-major (MATLAB order) with 0 = object, 1 = background.
        # Reshape in Fortran order to get upright images and scale {0, 1} -> {0, 255}.
        images = mat[f"{split}_data"].reshape(-1, 28, 28, order="F").astype(np.uint8) * 255
        # Labels are 1-based (1..101) in the source file.
        labels = mat[f"{split}_labels"].ravel().astype(np.int64) - 1

        for idx, (image, label) in enumerate(zip(images, labels)):
            yield idx, {"image": image, "label": int(label)}

    @staticmethod
    def _labels():
        """Class names in label order, verbatim from the source ``classnames`` field."""
        return [
            "Airplanes Side 2",
            "Faces 2",
            "Faces 3",
            "Leopards",
            "Motorbikes 16",
            "accordion",
            "anchor",
            "ant",
            "barrel",
            "bass",
            "beaver",
            "binocular",
            "bonsai",
            "brain",
            "brontosaurus",
            "buddha",
            "butterfly",
            "camera",
            "cannon",
            "car side",
            "ceiling fan",
            "cellphone",
            "chair",
            "chandelier",
            "cougar body",
            "cougar face",
            "crab",
            "crayfish",
            "crocodile",
            "crocodile head",
            "cup",
            "dalmatian",
            "dollar bill",
            "dolphin",
            "dragonfly",
            "electric guitar",
            "elephant",
            "emu",
            "euphonium",
            "ewer",
            "ferry",
            "flamingo",
            "flamingo head",
            "garfield",
            "gerenuk",
            "gramophone",
            "grand piano",
            "hawksbill",
            "headphone",
            "hedgehog",
            "helicopter",
            "ibis",
            "inline skate",
            "joshua tree",
            "kangaroo",
            "ketch",
            "lamp",
            "laptop",
            "llama",
            "lobster",
            "lotus",
            "mandolin",
            "mayfly",
            "menorah",
            "metronome",
            "minaret",
            "nautilus",
            "octopus",
            "okapi",
            "pagoda",
            "panda",
            "pigeon",
            "pizza",
            "platypus",
            "pyramid",
            "revolver",
            "rhino",
            "rooster",
            "saxophone",
            "schooner",
            "scissors",
            "scorpion",
            "sea horse",
            "snoopy",
            "soccer ball",
            "stapler",
            "starfish",
            "stegosaurus",
            "stop sign",
            "strawberry",
            "sunflower",
            "tick",
            "trilobite",
            "umbrella",
            "watch",
            "water lilly",
            "wheelchair",
            "wild cat",
            "windsor chair",
            "wrench",
            "yin yang",
        ]
