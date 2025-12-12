import datasets
import numpy as np

from stable_datasets.utils import BaseDatasetBuilder


MEDMNIST_VERSION = datasets.Version("1.0.0")


class MedMNISTConfig(datasets.BuilderConfig):
    """BuilderConfig with per-variant metadata used by MedMNIST._info()."""

    def __init__(self, *, num_classes: int, is_3d: bool = False, multi_label: bool = False, **kwargs):
        super().__init__(version=MEDMNIST_VERSION, **kwargs)
        self.num_classes = num_classes
        self.is_3d = is_3d
        self.multi_label = multi_label


class MedMNIST(BaseDatasetBuilder):
    """MedMNIST, a large-scale MNIST-like collection of standardized biomedical images, including 12 datasets for 2D and 6 datasets for 3D."""

    VERSION = MEDMNIST_VERSION

    BUILDER_CONFIGS = [
        # 2D Datasets
        MedMNISTConfig(name="pathmnist", description="MedMNIST PathMNIST (2D)", num_classes=9),
        MedMNISTConfig(
            name="chestmnist",
            description="MedMNIST ChestMNIST (2D, multi-label)",
            num_classes=14,
            multi_label=True,
        ),
        MedMNISTConfig(name="dermamnist", description="MedMNIST DermaMNIST (2D)", num_classes=7),
        MedMNISTConfig(name="octmnist", description="MedMNIST OCTMNIST (2D)", num_classes=4),
        MedMNISTConfig(name="pneumoniamnist", description="MedMNIST PneumoniaMNIST (2D)", num_classes=2),
        MedMNISTConfig(name="retinamnist", description="MedMNIST RetinaMNIST (2D)", num_classes=5),
        MedMNISTConfig(name="breastmnist", description="MedMNIST BreastMNIST (2D)", num_classes=2),
        MedMNISTConfig(name="bloodmnist", description="MedMNIST BloodMNIST (2D)", num_classes=8),
        MedMNISTConfig(name="tissuemnist", description="MedMNIST TissueMNIST (2D)", num_classes=8),
        MedMNISTConfig(name="organamnist", description="MedMNIST OrganAMNIST (2D)", num_classes=11),
        MedMNISTConfig(name="organcmnist", description="MedMNIST OrganCMNIST (2D)", num_classes=11),
        MedMNISTConfig(name="organsmnist", description="MedMNIST OrganSMNIST (2D)", num_classes=11),
        # 3D Datasets
        MedMNISTConfig(name="organmnist3d", description="MedMNIST OrganMNIST3D (3D)", num_classes=11, is_3d=True),
        MedMNISTConfig(name="nodulemnist3d", description="MedMNIST NoduleMNIST3D (3D)", num_classes=2, is_3d=True),
        MedMNISTConfig(name="adrenalmnist3d", description="MedMNIST AdrenalMNIST3D (3D)", num_classes=2, is_3d=True),
        MedMNISTConfig(name="fracturemnist3d", description="MedMNIST FractureMNIST3D (3D)", num_classes=3, is_3d=True),
        MedMNISTConfig(name="vesselmnist3d", description="MedMNIST VesselMNIST3D (3D)", num_classes=2, is_3d=True),
        MedMNISTConfig(name="synapsemnist3d", description="MedMNIST SynapseMNIST3D (3D)", num_classes=2, is_3d=True),
    ]

    def _source(self) -> dict:
        """Variant-aware source definition (computed from self.config at runtime)."""
        variant = self.config.name
        url = f"https://zenodo.org/records/10519652/files/{variant}.npz?download=1"
        # Single NPZ contains all splits; we map each split name to the same URL.
        return {
            "homepage": "https://medmnist.com/",
            "assets": {"train": url, "test": url, "val": url},
            "citation": """@article{medmnistv2,
                        title={MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image classification},
                        author={Yang, Jiancheng and Shi, Rui and Wei, Donglai and Liu, Zequan and Zhao, Lin and Ke, Bilian and Pfister, Hanspeter and Ni, Bingbing},
                        journal={Scientific Data},
                        volume={10},
                        number={1},
                        pages={41},
                        year={2023},
                        publisher={Nature Publishing Group UK London}
                    }""",
        }

    def _info(self):
        variant = self.config.name
        source = self._source()

        if getattr(self.config, "multi_label", False):  # multi-label instead of multi-class
            label_feature = datasets.Sequence(datasets.Value("int8"))
        else:
            label_feature = datasets.ClassLabel(num_classes=self.config.num_classes)

        return datasets.DatasetInfo(
            description=f"MedMNIST variant: {variant} dataset.",
            features=datasets.Features(
                {
                    "image": (
                        datasets.Array3D(shape=(28, 28, 28), dtype="uint8")
                        if getattr(self.config, "is_3d", False)
                        else datasets.Image()
                    ),
                    "label": label_feature,
                }
            ),
            supervised_keys=("image", "label"),
            homepage=source["homepage"],
            license="CC BY 4.0",
            citation=source["citation"],
        )

    def _generate_examples(self, data_path, split):
        data = np.load(data_path)
        images = data[f"{split}_images"]
        labels = data[f"{split}_labels"].squeeze()

        for idx, (image, label) in enumerate(zip(images, labels)):
            yield idx, {"image": image, "label": label}
