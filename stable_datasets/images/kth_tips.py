import io
import re
import tarfile
from pathlib import PurePosixPath

from PIL import Image as PILImage

from stable_datasets.schema import (
    ClassLabel,
    DatasetInfo,
    DatasetSource,
    DownloadInfo,
    Features,
    Value,
    Version,
)
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


class KTHTIPS(BaseDatasetBuilder):
    """KTH-TIPS material images under varying scale, pose, and lighting."""

    VERSION = Version("1.0.0")

    LABELS = [
        "aluminium_foil",
        "brown_bread",
        "corduroy",
        "cotton",
        "cracker",
        "linen",
        "orange_peel",
        "sandpaper",
        "sponge",
        "styrofoam",
    ]

    SOURCE = DatasetSource(
        homepage="https://www.csc.kth.se/cvap/databases/kth-tips/",
        license=(
            "Unspecified: no dataset-specific license identified in the official source. "
            "See the documentation for source links and usage information."
        ),
        citation="""@misc{fritz2004kthtips,
            title={The KTH-TIPS database},
            author={Fritz, Mario and Hayman, Eric and Caputo, Barbara and Eklundh, Jan-Olof},
            year={2004},
            url={https://www.csc.kth.se/cvap/databases/kth-tips/}
        }""",
        assets={
            "train": DownloadInfo(url="https://www.csc.kth.se/cvap/databases/kth-tips/kth_tips_col_200x200.tar"),
        },
    )

    def _info(self):
        return DatasetInfo(
            description="KTH-TIPS color images of 10 material classes.",
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=self.LABELS),
                    "scale": Value("int32"),
                    "image_id": Value("int32"),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
            license=self.SOURCE["license"],
        )

    def _generate_examples(self, data_path, split):
        with tarfile.open(data_path, "r:*") as archive:
            for member in archive:
                if not member.isfile():
                    continue

                path = PurePosixPath(member.name)
                label = path.parent.name
                match = re.fullmatch(
                    r"\d+-scale_(\d+)_im_(\d+)_col\.png",
                    path.name,
                )
                if label not in self.LABELS or match is None:
                    continue

                image_file = archive.extractfile(member)
                if image_file is None:
                    continue

                image_bytes = image_file.read()
                image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")

                yield (
                    member.name,
                    {
                        "image": image,
                        "label": label,
                        "scale": int(match.group(1)),
                        "image_id": int(match.group(2)),
                    },
                )
