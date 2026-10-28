import io
import re
import tarfile

from PIL import Image as PILImage

from stable_datasets.schema import (
    ClassLabel,
    DatasetInfo,
    DatasetSource,
    DownloadInfo,
    Features,
    Version,
)
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


class Chars74K(BaseDatasetBuilder):
    """Chars74K English handwritten character dataset."""

    VERSION = Version("1.0.0")

    LABELS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

    SOURCE = DatasetSource(
        homepage="https://teodecampos.github.io/chars74k/",
        assets={
            "train": DownloadInfo(url="https://info-ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishHnd.tgz"),
            "test": DownloadInfo(url="https://info-ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishHnd.tgz"),
        },
        citation="""@inproceedings{deCampos2009chars74k,
            title={Character Recognition in Natural Images},
            author={de Campos, Teofilo E. and Babu, Bodla Rakesh and Varma, Manik},
            booktitle={VISAPP},
            year={2009}
        }""",
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "Chars74K EnglishHnd contains 3,410 handwritten character "
                "images from 62 classes: digits, uppercase letters, and "
                "lowercase letters. This builder uses a deterministic "
                "80/20 train/test split."
            ),
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=self.LABELS),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        if split not in ("train", "test"):
            raise ValueError(f"Unsupported split: {split}")

        pattern = re.compile(r"^English/Hnd/Img/Sample(\d{3})/img(\d{3})-(\d{3})\.png$")

        with tarfile.open(data_path, "r:gz") as archive:
            members = sorted(
                (m for m in archive.getmembers() if m.isfile()),
                key=lambda m: m.name,
            )

            for member in members:
                if not member.name.lower().endswith(".png"):
                    continue

                match = pattern.fullmatch(member.name)
                if match is None:
                    raise ValueError(f"Unexpected image path: {member.name}")

                folder_id = int(match.group(1))
                filename_id = int(match.group(2))
                sample_id = int(match.group(3))

                if not (1 <= folder_id <= 62):
                    raise ValueError(f"Invalid class: {folder_id}")

                if filename_id != folder_id:
                    raise ValueError(f"Class mismatch: {member.name}")

                if not (1 <= sample_id <= 55):
                    raise ValueError(f"Invalid sample ID: {member.name}")

                is_test = (sample_id - 1) % 5 == 0
                if is_test != (split == "test"):
                    continue

                file = archive.extractfile(member)
                if file is None:
                    raise ValueError(f"Could not read image: {member.name}")

                with file:
                    image_bytes = file.read()

                with PILImage.open(io.BytesIO(image_bytes)) as img:
                    image = img.convert("RGB")

                label = folder_id - 1
                yield member.name, {"image": image, "label": label}
