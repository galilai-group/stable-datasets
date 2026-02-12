import io
import re
import tarfile

import datasets
from PIL import Image

from stable_datasets.utils import BaseDatasetBuilder


class FacePointing(BaseDatasetBuilder):
    """Head angle classification dataset."""

    VERSION = datasets.Version("1.0.0")

    # Single source-of-truth for dataset provenance + download locations.
    SOURCE = {
        "homepage": "http://crowley-coutaz.fr/HeadPoseDataSet/",
        "assets": {"train": "http://crowley-coutaz.fr/HeadPoseDataSet/HeadPoseImageDatabase.tar.gz"},
        "citation": """@inproceedings{gourier2004estimating,
                         title={Estimating face orientation from robust detection of salient facial features},
                         author={Gourier, Nicolas and Hall, Daniela and Crowley, James L},
                         booktitle={ICPR International Workshop on Visual Observation of Deictic Gestures},
                         year={2004},
                         organization={Citeseer}}""",
    }

    def _info(self):
        return datasets.DatasetInfo(
            description="""The head pose database consists of 15 sets of images. Each set contains 2 series of 93 images
                           of the same person at different poses. The database has a total size of approximately 30 MB.
                           Files are organized in directories, with each directory containing images from one person (2 series).
                           All images are in JPEG format. A Front directory contains 30 frontal images (pan and tilt angles equal to 0).""",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "person_id": datasets.Value("int32"),
                    "angles": datasets.Sequence(datasets.Value("int32")),
                }
            ),
            supervised_keys=("image", "angles"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        """Generate examples from the tar.gz archive.

        Each archive contains images with filenames encoding person_id and angles.
        Pattern: personXXYYY[+-]ZZ[+-]WW.jpg where XX=person_id, YYY=image_number, ZZ=tilt, WW=pan
        """
        with tarfile.open(data_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith(".jpg"):
                    # Match pattern: person(\d{2})\d+([+-]\d+)([+-]\d+)
                    match = re.search(r"person(\d{2})\d+([+-]\d+)([+-]\d+)", member.name)
                    if match:
                        person_id, tilt_angle, pan_angle = map(int, match.groups())
                        file = tar.extractfile(member)
                        image = Image.open(io.BytesIO(file.read())).convert("RGB")

                        yield (
                            f"{person_id}_{tilt_angle}_{pan_angle}_{member.name}",
                            {
                                "image": image,
                                "person_id": person_id,
                                "angles": [tilt_angle, pan_angle],
                            },
                        )
