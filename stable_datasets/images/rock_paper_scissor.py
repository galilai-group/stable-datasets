import io
import zipfile

from PIL import Image as PILImage

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


class RockPaperScissor(BaseDatasetBuilder):
    """Rock Paper Scissors dataset."""

    VERSION = Version("1.0.0")

    # Single source-of-truth for dataset provenance + download locations.
    SOURCE = DatasetSource(
        homepage="https://laurencemoroney.com/datasets.html",
        assets={
            "train": DownloadInfo(url="https://storage.googleapis.com/download.tensorflow.org/data/rps.zip"),
            "test": DownloadInfo(url="https://storage.googleapis.com/download.tensorflow.org/data/rps-test-set.zip"),
        },
        citation="""@misc{laurence2019rock,
                         title={Rock Paper Scissors Dataset},
                         author={Laurence Moroney},
                         year={2019},
                         url={https://laurencemoroney.com/datasets.html}}""",
        license="CC By 2.0",
    )

    def _info(self):
        return DatasetInfo(
            description="""Rock Paper Scissors contains images from various hands, from different races, ages, and
                           genders, posed into Rock / Paper or Scissors and labeled as such.""",
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=["rock", "paper", "scissors"]),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        """Yield examples by streaming PNGs directly from the zip archive."""
        with zipfile.ZipFile(data_path, "r") as archive:
            for name in archive.namelist():
                if not name.endswith(".png"):
                    continue
                # Label is the second-to-last path component
                # (e.g. 'rps/rock/rock01-000.png' -> 'rock').
                parts = name.rstrip("/").split("/")
                if len(parts) < 2:
                    continue
                label_name = parts[-2]
                # Source PNGs are RGBA (CGI renderer artifact); the underlying
                # content is 24-bit color per the dataset spec.
                image = PILImage.open(io.BytesIO(archive.read(name))).convert("RGB")
                yield name, {"image": image, "label": label_name}
