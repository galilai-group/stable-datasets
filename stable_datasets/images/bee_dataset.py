"""BeeDataset: top-down photos of honey bees with multi-label annotations."""

import io
import json
import zipfile

from PIL import Image as PILImage

from stable_datasets.schema import (
    DatasetInfo,
    DatasetSource,
    DownloadInfo,
    Features,
    Value,
    Version,
)
from stable_datasets.schema import (
    Image as ImageFeature,
)
from stable_datasets.utils import BaseDatasetBuilder


_LABELS = ("varroa", "pollen", "wasps", "cooling")
_IMAGE_DIR = "images_300"


class BeeDataset(BaseDatasetBuilder):
    """BeeDataset contains approximately 7,500 images with four binary labels.

    Images show bees or wasps from above against a green background.

    The labels are independent, so multiple labels may be active for one image:
    - varroa: varroa-mite infection
    - pollen: pollen packets are present
    - wasps: the image shows a wasp
    - cooling: cooling behavior is present

    The dataset has no official train/test split, so all examples are
    exposed through a single ``train`` split.

    Images in ``images_300`` have a height of 300 and a width of 150.
    """

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://github.com/BeeAlarmed/BeeDataset",
        assets={
            "train": DownloadInfo(
                url=("https://github.com/BeeAlarmed/BeeDataset/raw/refs/heads/main/BeeDataset_20201121.zip"),
            ),
        },
        citation="""@misc{hickert2021beedataset,
            title={Dataset for a camera based bee-hive monitoring},
            author={Fabian Hickert},
            year={2021},
            url={https://github.com/BeeAlarmed/BeeDataset}
        }""",
        license="GPL-3.0",
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "BeeDataset contains images of bees and wasps photographed "
                "from above at a hive entrance. Each image has four "
                "independent binary labels: varroa-mite infection, pollen "
                "packets, wasps, and cooling behavior."
            ),
            features=Features(
                {
                    "image": ImageFeature(),
                    **{label: Value("int32") for label in _LABELS},
                }
            ),
            supervised_keys=None,
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
            license=self.SOURCE["license"],
        )

    def _generate_examples(self, data_path, split):
        """Yield images and their multi-label annotations from the ZIP archive."""
        if split != "train":
            raise ValueError(f"Unknown BeeDataset split: {split!r}")

        with zipfile.ZipFile(data_path) as archive:
            json_members = [
                name
                for name in archive.namelist()
                if name.rsplit("/", 1)[-1] == "data.json" and "__MACOSX" not in name.split("/")
            ]

            if len(json_members) != 1:
                raise ValueError(
                    f"Expected exactly one data.json in the archive, found {len(json_members)}: {json_members}"
                )

            json_member = json_members[0]
            root = json_member[: -len("data.json")]
            annotations = json.loads(archive.read(json_member))

            # Sorting gives a reproducible example order.
            for name in sorted(annotations):
                entry = annotations[name]

                missing_labels = [label for label in _LABELS if label not in entry]
                if missing_labels:
                    raise ValueError(f"Missing labels for image {name!r}: {missing_labels}")

                labels = {label: int(bool(entry[label])) for label in _LABELS}

                image_path = f"{root}{_IMAGE_DIR}/{name}"
                try:
                    image_bytes = archive.read(image_path)
                except KeyError as exc:
                    raise ValueError(
                        f"Image {image_path!r} referenced by data.json was not found in the archive."
                    ) from exc

                with PILImage.open(io.BytesIO(image_bytes)) as img:
                    if img.mode == "RGB":
                        image = image_bytes
                    else:
                        image = img.convert("RGB").copy()

                yield name, {"image": image, **labels}
