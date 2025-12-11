import datasets
import importlib.util
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

_IN10_classes = [
    "n01440764",
    "n02102040",
    "n02979186",
    "n03000684",
    "n03028079",
    "n03394916",
    "n03417042",
    "n03425413",
    "n03445777",
    "n03888257",
]
_IN100_CLASSES = [
    "n02869837",
    "n01749939",
    "n02488291",
    "n02107142",
    "n13037406",
    "n02091831",
    "n04517823",
    "n04589890",
    "n03062245",
    "n01773797",
    "n01735189",
    "n07831146",
    "n07753275",
    "n03085013",
    "n04485082",
    "n02105505",
    "n01983481",
    "n02788148",
    "n03530642",
    "n04435653",
    "n02086910",
    "n02859443",
    "n13040303",
    "n03594734",
    "n02085620",
    "n02099849",
    "n01558993",
    "n04493381",
    "n02109047",
    "n04111531",
    "n02877765",
    "n04429376",
    "n02009229",
    "n01978455",
    "n02106550",
    "n01820546",
    "n01692333",
    "n07714571",
    "n02974003",
    "n02114855",
    "n03785016",
    "n03764736",
    "n03775546",
    "n02087046",
    "n07836838",
    "n04099969",
    "n04592741",
    "n03891251",
    "n02701002",
    "n03379051",
    "n02259212",
    "n07715103",
    "n03947888",
    "n04026417",
    "n02326432",
    "n03637318",
    "n01980166",
    "n02113799",
    "n02086240",
    "n03903868",
    "n02483362",
    "n04127249",
    "n02089973",
    "n03017168",
    "n02093428",
    "n02804414",
    "n02396427",
    "n04418357",
    "n02172182",
    "n01729322",
    "n02113978",
    "n03787032",
    "n02089867",
    "n02119022",
    "n03777754",
    "n04238763",
    "n02231487",
    "n03032252",
    "n02138441",
    "n02104029",
    "n03837869",
    "n03494278",
    "n04136333",
    "n03794056",
    "n03492542",
    "n02018207",
    "n04067472",
    "n03930630",
    "n03584829",
    "n02123045",
    "n04229816",
    "n02100583",
    "n03642806",
    "n04336792",
    "n03259280",
    "n02116738",
    "n02108089",
    "n03424325",
    "n01855672",
    "n02090622",
]


class Imagenette(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.1.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="imagenet",
            version=VERSION,
            description="1000-class version",
        ),
        datasets.BuilderConfig(
            name="imagenette",
            version=VERSION,
            description="10-class version",
        ),
        datasets.BuilderConfig(
            name="imagenet100",
            version=VERSION,
            description="100-class version",
        ),
    ]

    DEFAULT_CONFIG_NAME = "imagenette"

    def _get_in1k_module(self):
        if hasattr(self, "_in_mod"):
            return self._in_mod
        path = Path(
            snapshot_download(repo_id="ILSVRC/imagenet-1k", repo_type="dataset")
        )
        print(path)
        if not (path / "imagenet_1k.py").is_file():
            (path / "imagenet-1k.py").rename(path / "imagenet_1k.py")
        if not (path / "__init__.py").is_file():
            (path / "__init__.py").touch()
            (path / "__init__.py").write_text("from . import imagenet_1k")

        sys.path.append(str(path.parent))
        self._in_mod = importlib.import_module(path.name)
        self._DATA_URL = path
        return self._in_mod

    def _info(self):
        if self.config.name == "imagenet":
            mod = self._get_in1k_module()
            return mod.imagenet_1k.Imagenet1k()._info()
        elif self.config.name == "imagenet100":
            names = _IN100_CLASSES
        else:
            names = _IN10_classes
        features = datasets.Features(
            {
                "image": datasets.Image(),
                "label": datasets.ClassLabel(names=names),
            }
        )

        if self.config.name == "imagenet":
            homepage = "https://www.image-net.org/update-mar-11-2021.php"
            license = "CC BY 2.0"
        elif self.config.name == "imagenette":
            homepage = "https://github.com/fastai/imagenette"
            license = "Apache 2.0"

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description="Imagenet and its variants",
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            homepage=homepage,
            license=license,
            # Citation for the dataset
            # citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        if self.config.name == "imagenet":
            mod = self._get_in1k_module()
            # print(self._DATA_URL, mod.imagenet_1k._DATA_URL)
            mod.imagenet_1k._DATA_URL = {
                fold: [(self._DATA_URL / p) for p in mod.imagenet_1k._DATA_URL[fold]]
                for fold in mod.imagenet_1k._DATA_URL
            }
            print(mod.imagenet_1k._DATA_URL)
            archives = dl_manager.download(mod.imagenet_1k._DATA_URL)

            print(archives)

            print([dl_manager.iter_archive(archive) for archive in archives["train"]])

            print(
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "archives": [
                            dl_manager.iter_archive(archive)
                            for archive in archives["train"]
                        ],
                        "split": "train",
                    },
                )
            )

            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "archives": [
                            dl_manager.iter_archive(archive)
                            for archive in archives["train"]
                        ],
                        "split": "train",
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "archives": [
                            dl_manager.iter_archive(archive)
                            for archive in archives["val"]
                        ],
                        "split": "validation",
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "archives": [
                            dl_manager.iter_archive(archive)
                            for archive in archives["test"]
                        ],
                        "split": "test",
                    },
                ),
            ]
            # return mod.imagenet_1k.Imagenet1k()._split_generators(dl_manager)
        elif self.config.name == "imagenette":
            urls = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
        elif self.config.name == "imagenet100":
            d = datasets.load_dataset("imagenet-1k")
            d["train"] = d["train"].filter(
                lambda example: example["label"] in _IN100_CLASSES
            )
            d["validation"] = d["validation"].filter(
                lambda example: example["label"] in _IN100_CLASSES
            )
        data_dir = Path(dl_manager.download_and_extract(urls))
        train_path = data_dir / "imagenette2" / "train"
        test_path = data_dir / "imagenette2" / "val"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"files": train_path.rglob("*.JPEG")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"files": test_path.rglob("*.JPEG")},
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, **kwargs):
        if self.config.name == "imagenet":
            mod = self._get_in1k_module()
            return mod.imagenet_1k.Imagenet1k()._generate_examples(**kwargs)
        files = kwargs["files"]
        for key, file in enumerate(files):
            image = str(file)  # Image.open(file).convert("RGB")
            if self.config.name == "imagenette":
                label = file.parent.name
            yield key, {"image": image, "label": label}
