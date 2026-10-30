"""Omniglot images with the original background/evaluation partition."""

import re
import zipfile
from pathlib import PurePosixPath

from PIL import Image as PILImage

from stable_datasets.schema import (
    ClassLabel,
    DatasetInfo,
    DatasetSource,
    DownloadInfo,
    Features,
    Image,
    Value,
    Version,
)
from stable_datasets.utils import BaseDatasetBuilder


_BASE_URL = "https://raw.githubusercontent.com/brendenlake/omniglot/057f034baf2ecb8530bc5710e5a23584d2a519cc/python/"


class Omniglot(BaseDatasetBuilder):
    """Load 1,623 character classes from 50 alphabets, with 20 drawings each.

    Background and evaluation contain disjoint character classes. Class IDs
    are the released filename prefixes minus one, shared across both splits.
    Images retain their 105 x 105 pixels and black-ink/white-paper polarity.
    """

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="https://github.com/brendenlake/omniglot",
        license="MIT",
        citation="""@article{lake2015human,
  title={Human-level concept learning through probabilistic program induction},
  author={Brenden M. Lake and Ruslan Salakhutdinov and Joshua B. Tenenbaum},
  journal={Science}, volume={350}, number={6266}, pages={1332--1338}, year={2015},
  doi={10.1126/science.aab3050}
}""",
        assets={
            "background": DownloadInfo(
                url=_BASE_URL + "images_background.zip",
                checksum="sha256:ad41ab679c8b5d90b271ef46be6987cc81211774a695c29dcc5367b2b26ee640",
            ),
            "evaluation": DownloadInfo(
                url=_BASE_URL + "images_evaluation.zip",
                checksum="sha256:1f61a8f3366785b057fc117d9228e78a16e3d976c8953b2a10fcc74cf0609cee",
            ),
        },
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "Omniglot handwritten characters: 964 background classes from "
                "30 alphabets and 659 evaluation classes from 20 alphabets. "
                "The evaluation classes are unseen during background learning."
            ),
            features=Features(
                {
                    "image": Image(),
                    "label": ClassLabel(num_classes=1623),
                    "alphabet": Value("string"),
                    "character": Value("string"),
                    "example_id": Value("int32"),
                    "filename": Value("string"),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE.homepage,
            citation=self.SOURCE.citation,
            license=self.SOURCE.license,
        )

    def _generate_examples(self, data_path, split):
        if split not in ("background", "evaluation"):
            raise ValueError(f"Unknown Omniglot split: {split!r}")
        root = f"images_{split}"
        with zipfile.ZipFile(data_path) as archive:
            names = sorted(
                name for name in archive.namelist() if name.startswith(root + "/") and name.endswith(".png")
            )
            if not names:
                raise ValueError(f"No Omniglot PNG images found under {root}/")
            for name in names:
                path = PurePosixPath(name)
                if len(path.parts) != 4:
                    raise ValueError(f"Unexpected Omniglot image path: {name}")
                _, alphabet, character, filename = path.parts
                match = re.fullmatch(r"(\d{4})_(\d{2})\.png", filename)
                if match is None:
                    raise ValueError(f"Unexpected Omniglot filename: {filename}")
                character_id, example_id = map(int, match.groups())
                lower, upper = (1, 964) if split == "background" else (965, 1623)
                if not lower <= character_id <= upper or not 1 <= example_id <= 20:
                    raise ValueError(f"Omniglot character/example ID is outside the {split} range: {filename}")
                with archive.open(name) as handle, PILImage.open(handle) as original:
                    if original.size != (105, 105):
                        raise ValueError(f"Expected 105 x 105 Omniglot image: {name}")
                    # Released PNGs use mode '1'. L preserves their black/white
                    # values as uint8 0/255, including correct torch formatting.
                    image = original.convert("L")
                yield (
                    name,
                    {
                        "image": image,
                        "label": character_id - 1,
                        "alphabet": alphabet,
                        "character": f"{alphabet}/{character}",
                        "example_id": example_id,
                        "filename": f"{alphabet}/{character}/{filename}",
                    },
                )
