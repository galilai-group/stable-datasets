import datasets
import numpy as np

from stable_datasets.utils import BaseDatasetBuilder


MEDMNIST_VERSION = datasets.Version("1.0.0")

_VALID_SIZES_2D = (28, 64, 128, 224)
_VALID_SIZES_3D = (28, 64)


class MedMNISTConfig(datasets.BuilderConfig):
    """BuilderConfig with per-variant metadata used by MedMNIST._info().

    Args:
        num_classes: Number of target classes.
        is_3d: Whether the variant is a 3D volumetric dataset.
        multi_label: Whether the task is multi-label classification.
        size: Image resolution. 2D datasets support 28, 64, 128, 224;
              3D datasets support 28, 64. Defaults to 28 (MNIST-like).
    """

    def __init__(
        self,
        *,
        num_classes: int,
        is_3d: bool = False,
        multi_label: bool = False,
        size: int = 28,
        **kwargs,
    ):
        super().__init__(version=MEDMNIST_VERSION, **kwargs)
        valid_sizes = _VALID_SIZES_3D if is_3d else _VALID_SIZES_2D
        if size not in valid_sizes:
            raise ValueError(
                f"size={size} is not valid for {'3D' if is_3d else '2D'} variant "
                f"'{kwargs.get('name', '?')}'. Choose from {valid_sizes}."
            )
        self.num_classes = num_classes
        self.is_3d = is_3d
        self.multi_label = multi_label
        self.size = size


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
        """Variant- and size-aware source definition."""
        variant = self.config.name
        size = getattr(self.config, "size", 28)
        is_3d = getattr(self.config, "is_3d", False)

        valid_sizes = _VALID_SIZES_3D if is_3d else _VALID_SIZES_2D
        if size not in valid_sizes:
            raise ValueError(
                f"size={size} is not valid for {'3D' if is_3d else '2D'} variant "
                f"'{variant}'. Choose from {valid_sizes}."
            )

        filename = f"{variant}.npz" if size == 28 else f"{variant}_{size}.npz"
        url = f"https://zenodo.org/records/10519652/files/{filename}?download=1"

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
        size = getattr(self.config, "size", 28)
        source = self._source()

        if getattr(self.config, "multi_label", False):
            label_feature = datasets.Sequence(datasets.Value("int8"))
        else:
            label_feature = datasets.ClassLabel(num_classes=self.config.num_classes)

        if getattr(self.config, "is_3d", False):
            image_feature = datasets.Array3D(shape=(size, size, size), dtype="uint8")
        else:
            image_feature = datasets.Image()

        return datasets.DatasetInfo(
            description=f"MedMNIST variant: {variant} (size={size}).",
            features=datasets.Features(
                {
                    "image": image_feature,
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
