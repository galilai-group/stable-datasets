import io
import tarfile
from pathlib import Path

import datasets
from PIL import Image as PILImage

from stable_datasets.utils import BaseDatasetBuilder, _default_dest_folder, bulk_download


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
    """Imagenette (ImageNet-10) from FastAI's public tarball."""

    VERSION = datasets.Version("2.0.0")

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

    def _info(self):
        return datasets.DatasetInfo(
            description="ImageNet-10 (Imagenette) with train/validation splits.",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(names=_IN10_CLASSES),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _split_generators(self, dl_manager):
        source = self._source()
        assets = source["assets"]

        urls = list(assets.values())
        download_dir = getattr(self, "_raw_download_dir", None)
        if download_dir is None:
            download_dir = _default_dest_folder()

        downloaded_paths = bulk_download(urls, dest_folder=download_dir)
        archive_path = downloaded_paths[0]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_path": archive_path, "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"data_path": archive_path, "split": "val"},
            ),
        ]

    def _generate_examples(self, data_path, split):
        with tarfile.open(Path(data_path), "r:*") as archive:
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
