import tarfile
from pathlib import Path

from stable_datasets.schema import ClassLabel, DatasetInfo, Features, Image, Version
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, download


_IN10_CLASSES = [
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


class Imagenette(BaseDatasetBuilder):
    """Imagenette dataset — a 10-class subset of ImageNet for quick benchmarking."""

    VERSION = Version("1.1.0")

    SOURCE = {
        "homepage": "https://github.com/fastai/imagenette",
        "citation": """@misc{howard2019imagenette,
                         author={Jeremy Howard},
                         title={Imagenette: A smaller subset of 10 easily classified classes from Imagenet},
                         year={2019},
                         url={https://github.com/fastai/imagenette}}""",
        "assets": {
            "imagenette": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
        },
    }

    def _info(self):
        features = Features(
            {
                "image": Image(),
                "label": ClassLabel(names=_IN10_CLASSES),
            }
        )

        return DatasetInfo(
            description="Imagenette: a 10-class subset of ImageNet for fast prototyping.",
            features=features,
            homepage=self.SOURCE["homepage"],
            license="Apache 2.0",
            citation=self.SOURCE["citation"],
        )

    def _split_generators(self, dl_manager=None):
        source = self._source()
        url = source["assets"]["imagenette"]
        archive_path = download(url, dest_folder=self._raw_download_dir)

        # Extract the tarball
        extract_dir = Path(self._raw_download_dir) / "imagenette-extracted"
        if not extract_dir.exists():
            extract_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(extract_dir)

        train_path = extract_dir / "imagenette2" / "train"
        test_path = extract_dir / "imagenette2" / "val"

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"data_dir": str(train_path)},
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={"data_dir": str(test_path)},
            ),
        ]

    def _generate_examples(self, data_dir):
        data_path = Path(data_dir)
        for key, file in enumerate(sorted(data_path.rglob("*.JPEG"))):
            label = file.parent.name
            yield key, {"image": str(file), "label": label}
