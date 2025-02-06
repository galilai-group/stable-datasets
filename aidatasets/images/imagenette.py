import datasets
from pathlib import Path


class Imagenette(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.1.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="A smaller subset of 10 easily classified classes from Imagenet.",
            features=datasets.Features({
                "image": datasets.Image(),
                "label": datasets.ClassLabel(names=self._class_names()),
            }),
            supervised_keys=("image", "label"),
            homepage="https://github.com/fastai/imagenette",
            license="Apache 2.0",
        )

    def _split_generators(self, dl_manager):
        urls = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
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

    def _generate_examples(self, files):
        for key, file in enumerate(files):
            image = str(file)
            label = file.parent.name
            yield key, {"image": image, "label": label}

    @staticmethod
    def _class_names():
        return [
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
