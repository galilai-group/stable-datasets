"""MHEALTH continuous recordings with all channels and sample-level labels."""

import zipfile

import numpy as np

from stable_datasets.schema import (
    ClassLabel,
    DatasetInfo,
    DatasetSource,
    DownloadInfo,
    Features,
    Sequence,
    Value,
    Version,
)
from stable_datasets.utils import BaseDatasetBuilder


class MHEALTH(BaseDatasetBuilder):
    """One (time, 23-channel) recording per subject in a single 'train' split."""

    VERSION = Version("1.0.0")
    SAMPLE_RATE = 50
    CHANNEL_NAMES = (
        "chest_acc_x",
        "chest_acc_y",
        "chest_acc_z",
        "ecg_1",
        "ecg_2",
        "ankle_acc_x",
        "ankle_acc_y",
        "ankle_acc_z",
        "ankle_gyro_x",
        "ankle_gyro_y",
        "ankle_gyro_z",
        "ankle_mag_x",
        "ankle_mag_y",
        "ankle_mag_z",
        "arm_acc_x",
        "arm_acc_y",
        "arm_acc_z",
        "arm_gyro_x",
        "arm_gyro_y",
        "arm_gyro_z",
        "arm_mag_x",
        "arm_mag_y",
        "arm_mag_z",
    )
    SOURCE = DatasetSource(
        homepage="https://archive.ics.uci.edu/dataset/319/mhealth+dataset",
        assets={
            "train": DownloadInfo(
                url="https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip",
                filename="MHEALTH.zip",
                checksum="sha256:16ad0ce709f3f00df18f348610d15bce0884b79e2143f57f446493673f02b8e0",
            ),
        },
        citation="""@misc{mhealth_2014,
  author = {Banos, Oresti and Garcia, Rafael and Saez, Alejandro},
  title = {MHEALTH},
  year = {2014},
  publisher = {UCI Machine Learning Repository},
  doi = {10.24432/C5TW22},
  url = {https://doi.org/10.24432/C5TW22}
}""",
        license="CC BY 4.0",
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "MHEALTH: ten continuous subject recordings at 50 Hz, with all 23 signal channels "
                "including both ECG leads, aligned activity labels 0-12, and original subject IDs. "
                "Label 0 is the null class. No windowing, normalization, filtering, or subject split is applied."
            ),
            features=Features(
                {
                    "series": Sequence(Sequence(Value("float32"))),
                    "label": Sequence(
                        ClassLabel(
                            names=[
                                "null",
                                "standing_still",
                                "sitting_relaxing",
                                "lying_down",
                                "walking",
                                "climbing_stairs",
                                "waist_bends_forward",
                                "frontal_elevation_arms",
                                "knees_bending",
                                "cycling",
                                "jogging",
                                "running",
                                "jump_front_back",
                            ]
                        )
                    ),
                    "subject_id": Value("int32"),
                }
            ),
            supervised_keys=("series", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
            license=self.SOURCE["license"],
        )

    def _generate_examples(self, data_path, split):
        del split
        with zipfile.ZipFile(data_path) as archive:
            for subject in range(1, 11):
                name = f"MHEALTHDATASET/mHealth_subject{subject}.log"
                with archive.open(name) as stream:
                    values = np.loadtxt(stream, dtype=np.float32, ndmin=2)
                if values.shape[0] == 0 or values.shape[1] != 24:
                    raise ValueError(f"MHEALTH {name} must have 24 columns and at least one row.")
                if not np.isfinite(values).all():
                    raise ValueError(f"MHEALTH {name} must contain finite values.")
                labels = values[:, 23]
                if np.any((labels < 0) | (labels > 12) | (labels != np.floor(labels))):
                    raise ValueError(f"MHEALTH {name} activity labels must be integers in 0-12.")
                yield (
                    subject,
                    {
                        "series": values[:, :23],
                        "label": labels.astype(np.int64),
                        "subject_id": subject,
                    },
                )
