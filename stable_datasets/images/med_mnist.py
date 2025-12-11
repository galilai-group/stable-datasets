import datasets
import numpy as np


class MedMNISTConfig(datasets.BuilderConfig):
    def __init__(self, variant, **kwargs):
        super(MedMNISTConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)
        self.variant = variant


class MedMNIST(datasets.GeneratorBasedBuilder):
    """MedMNIST, a large-scale MNIST-like collection of standardized biomedical images, including 12 datasets for 2D and 6 datasets for 3D.
    """
    BUILDER_CONFIGS = [
        # 2D Datasets
        MedMNISTConfig(name="pathmnist", variant="pathmnist"),
        MedMNISTConfig(name="chestmnist", variant="chestmnist"),
        MedMNISTConfig(name="dermamnist", variant="dermamnist"),
        MedMNISTConfig(name="octmnist", variant="octmnist"),
        MedMNISTConfig(name="pneumoniamnist", variant="pneumoniamnist"),
        MedMNISTConfig(name="retinamnist", variant="retinamnist"),
        MedMNISTConfig(name="breastmnist", variant="breastmnist"),
        MedMNISTConfig(name="bloodmnist", variant="bloodmnist"),
        MedMNISTConfig(name="tissuemnist", variant="tissuemnist"),
        MedMNISTConfig(name="organamnist", variant="organamnist"),
        MedMNISTConfig(name="organcmnist", variant="organcmnist"),
        MedMNISTConfig(name="organsmnist", variant="organsmnist"),
        # 3D Datasets
        MedMNISTConfig(name="organmnist3d", variant="organmnist3d"),
        MedMNISTConfig(name="nodulemnist3d", variant="nodulemnist3d"),
        MedMNISTConfig(name="adrenalmnist3d", variant="adrenalmnist3d"),
        MedMNISTConfig(name="fracturemnist3d", variant="fracturemnist3d"),
        MedMNISTConfig(name="vesselmnist3d", variant="vesselmnist3d"),
        MedMNISTConfig(name="synapsemnist3d", variant="synapsemnist3d"),
    ]

    def _info(self):
        variant = self.config.variant
        num_classes = {
            "pathmnist": 9,
            "chestmnist": 14,
            "dermamnist": 7,
            "octmnist": 4,
            "pneumoniamnist": 2,
            "retinamnist": 5,
            "breastmnist": 2,
            "bloodmnist": 8,
            "tissuemnist": 8,
            "organamnist": 11,
            "organcmnist": 11,
            "organsmnist": 11,
            "organmnist3d": 11,
            "nodulemnist3d": 2,
            "adrenalmnist3d": 2,
            "fracturemnist3d": 3,
            "vesselmnist3d": 2,
            "synapsemnist3d": 2,
        }.get(variant, 0)

        if variant == "chestmnist":     # multi-label instead of multi-class
            label_feature = datasets.Sequence(datasets.Value("int8"))
        else:
            label_feature = datasets.ClassLabel(num_classes=num_classes)

        return datasets.DatasetInfo(
            description=f"MedMNIST variant: {variant} dataset.",
            features=datasets.Features(
                {
                    "image": datasets.Array3D(shape=(28, 28, 28), dtype="uint8")
                    if '3d' in variant else datasets.Image(),
                    "label": label_feature,
                }
            ),
            supervised_keys=("image", "label"),
            homepage="https://medmnist.com/",
            license="CC BY 4.0",
            citation="""@article{medmnistv2,
                        title={MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image classification},
                        author={Yang, Jiancheng and Shi, Rui and Wei, Donglai and Liu, Zequan and Zhao, Lin and Ke, Bilian and Pfister, Hanspeter and Ni, Bingbing},
                        journal={Scientific Data},
                        volume={10},
                        number={1},
                        pages={41},
                        year={2023},
                        publisher={Nature Publishing Group UK London}
                    }""",
        )

    def _split_generators(self, dl_manager):
        variant = self.config.variant
        url = f"https://zenodo.org/records/10519652/files/{variant}.npz?download=1"
        file_path = dl_manager.download(url)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"file_path": file_path, "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"file_path": file_path, "split": "test"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"file_path": file_path, "split": "val"},
            ),
        ]

    def _generate_examples(self, file_path, split):
        data = np.load(file_path)
        images = data[f"{split}_images"]
        labels = data[f"{split}_labels"].squeeze()

        for idx, (image, label) in enumerate(zip(images, labels)):
            yield idx, {"image": image, "label": label}
