import io
import tarfile
from pathlib import Path

from PIL import Image as PILImage

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
    """Imagenette: 10 easily classified classes from ImageNet."""

    VERSION = Version("2.0.0")
    SOURCE = {
        "homepage": "https://github.com/fastai/imagenette",
        "assets": {
            "archive": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
        },
        "citation": """@misc{howard2019imagenette,
            author={Jeremy Howard},
            title={Imagenette: A smaller subset of 10 easily classified classes from ImageNet},
            year={2019},
            url={https://github.com/fastai/imagenette}
        }""",
    }

    def __init__(self, streaming: bool = False, **kwargs):
        self.streaming = streaming
        super().__init__(**kwargs)

    def _info(self):
        return DatasetInfo(
            description="Imagenette with train/validation splits.",
            features=Features({"image": Image(), "label": ClassLabel(names=_IN10_CLASSES)}),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _split_generators(self):
        archive_path = download(self.SOURCE["assets"]["archive"], dest_folder=self._raw_download_dir)
        return [
            SplitGenerator(name=Split.TRAIN, gen_kwargs={"data_path": archive_path, "split": "train"}),
            SplitGenerator(name=Split.TEST, gen_kwargs={"data_path": archive_path, "split": "val"}),
        ]

    def _generate_examples(self, data_path, split):
        mode = "r|*" if self.streaming else "r:*"
        with tarfile.open(Path(data_path), mode) as archive:
            for member in archive:
                if not member.isfile() or not member.name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                parts = member.name.split("/")
                if len(parts) < 4:
                    continue
                if parts[0] != "imagenette2" or parts[1] != split:
                    continue
                wnid = parts[2]
                if wnid not in _IN10_CLASSES:
                    continue

                file_obj = archive.extractfile(member)
                if file_obj is None:
                    continue
                image = PILImage.open(io.BytesIO(file_obj.read())).convert("RGB")
                label = _IN10_CLASSES.index(wnid)
                yield member.name, {"image": image, "label": label}
