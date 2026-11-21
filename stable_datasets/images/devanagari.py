from zipfile import ZipFile

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Image, Version
from stable_datasets.utils import BaseDatasetBuilder


class Devanagari(BaseDatasetBuilder):
    """DHCD: 46 handwritten character classes with the original train/test split."""

    VERSION = Version("1.0.0")
    # Preserve the archive's numeric character order, followed by digits 0-9.
    CLASS_NAMES = (
        "character_1_ka",
        "character_2_kha",
        "character_3_ga",
        "character_4_gha",
        "character_5_kna",
        "character_6_cha",
        "character_7_chha",
        "character_8_ja",
        "character_9_jha",
        "character_10_yna",
        "character_11_taamatar",
        "character_12_thaa",
        "character_13_daa",
        "character_14_dhaa",
        "character_15_adna",
        "character_16_tabala",
        "character_17_tha",
        "character_18_da",
        "character_19_dha",
        "character_20_na",
        "character_21_pa",
        "character_22_pha",
        "character_23_ba",
        "character_24_bha",
        "character_25_ma",
        "character_26_yaw",
        "character_27_ra",
        "character_28_la",
        "character_29_waw",
        "character_30_motosaw",
        "character_31_petchiryakha",
        "character_32_patalosaw",
        "character_33_ha",
        "character_34_chhya",
        "character_35_tra",
        "character_36_gya",
        "digit_0",
        "digit_1",
        "digit_2",
        "digit_3",
        "digit_4",
        "digit_5",
        "digit_6",
        "digit_7",
        "digit_8",
        "digit_9",
    )
    SOURCE = DatasetSource(
        homepage="https://archive.ics.uci.edu/dataset/389/devanagari+handwritten+character+dataset",
        assets={
            split: DownloadInfo(
                url="https://archive.ics.uci.edu/static/public/389/devanagari+handwritten+character+dataset.zip",
                checksum="sha256:752e6a2e6b0d1b475375e6a9db937918bbbf7c6b7d137dbe234264029fd3b88d",
            )
            for split in ("train", "test")
        },
        license="CC BY 4.0",
        citation="""@misc{dhcd2015,
            author={Acharya, Shailesh and Gyawali, Prashnna},
            title={Devanagari Handwritten Character Dataset},
            year={2015}, publisher={UCI Machine Learning Repository},
            doi={10.24432/C5XS53}}
        """,
    )

    def _info(self):
        return DatasetInfo(
            description="92,000 32x32 grayscale images of 36 Devanagari characters and 10 digits.",
            features=Features({"image": Image(), "label": ClassLabel(names=list(self.CLASS_NAMES))}),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE.homepage,
            citation=self.SOURCE.citation,
            license=self.SOURCE.license,
        )

    def _generate_examples(self, data_path, split):
        if split not in ("train", "test"):
            raise ValueError(f"Unknown Devanagari split: {split!r}")
        prefix = f"DevanagariHandwrittenCharacterDataset/{split.title()}/"
        labels = {name: index for index, name in enumerate(self.CLASS_NAMES)}
        with ZipFile(data_path) as archive:
            members = sorted(name for name in archive.namelist() if name.startswith(prefix) and name.endswith(".png"))
            if not members:
                raise ValueError(f"No images found for Devanagari split {split!r}")
            for name in members:
                relative = name[len(prefix) :].split("/")
                if len(relative) != 2 or relative[0] not in labels:
                    raise ValueError(f"Unexpected Devanagari image path: {name!r}")
                # Keep the original PNG bytes, including their grayscale encoding.
                yield name, {"image": archive.read(name), "label": labels[relative[0]]}
