import zipfile
from pathlib import PurePosixPath

from PIL import Image as PILImage
from tqdm import tqdm

from stable_datasets.schema import ClassLabel, DatasetInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


class RESISC45(BaseDatasetBuilder):
    """NWPU-RESISC45: 31,500 RGB images in 45 remote-sensing scene categories.

    The complete collection is exposed as ``train``. No fixed official holdout
    is supplied; partitions introduced by later papers are not applied.
    """

    VERSION = Version("1.0.0")

    SOURCE = {
        "homepage": "http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html",
        "assets": {
            "train": (
                "https://hf.co/datasets/isaaccorley/resisc45/resolve/"
                "883edc0eee77b2c84225472f10f126e3ed83fa6e/NWPU-RESISC45.zip"
            ),
        },
        "mirror": "https://huggingface.co/datasets/isaaccorley/resisc45",
        "citation": """@article{cheng2017remote,
            title={Remote Sensing Image Scene Classification: Benchmark and State of the Art},
            author={Cheng, Gong and Han, Junwei and Lu, Xiaoqiang},
            journal={Proceedings of the IEEE},
            volume={105},
            number={10},
            pages={1865--1883},
            year={2017},
            doi={10.1109/JPROC.2017.2675998}
        }""",
    }

    def _info(self):
        return DatasetInfo(
            description=(
                "NWPU-RESISC45: 31,500 remote-sensing scene images in 45 classes at their native "
                "256x256 resolution. The full collection is exposed as train, without an official fixed holdout."
            ),
            features=Features({"image": ImageFeature(), "label": ClassLabel(names=self._labels())}),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        class_to_label = {name: index for index, name in enumerate(self._labels())}
        found_image = False
        with zipfile.ZipFile(data_path) as archive:
            for entry in tqdm(
                sorted(archive.infolist(), key=lambda item: item.filename), desc=f"Processing {split} set"
            ):
                path = PurePosixPath(entry.filename)
                if entry.is_dir() or path.suffix.lower() != ".jpg":
                    continue
                if len(path.parts) != 3 or path.parts[0] != "NWPU-RESISC45":
                    continue
                class_name = path.parent.name
                if class_name not in class_to_label:
                    raise ValueError(f"Unknown RESISC45 class {class_name!r} in {entry.filename!r}.")
                with archive.open(entry) as image_file:
                    with PILImage.open(image_file) as source_image:
                        image = source_image.convert("RGB")
                found_image = True
                yield entry.filename, {"image": image, "label": class_to_label[class_name]}
        if not found_image:
            raise ValueError("No RESISC45 images found under NWPU-RESISC45/<class>/ in the archive.")

    @staticmethod
    def _labels():
        """Original class-directory names, in fixed alphabetical order."""
        return [
            "airplane",
            "airport",
            "baseball_diamond",
            "basketball_court",
            "beach",
            "bridge",
            "chaparral",
            "church",
            "circular_farmland",
            "cloud",
            "commercial_area",
            "dense_residential",
            "desert",
            "forest",
            "freeway",
            "golf_course",
            "ground_track_field",
            "harbor",
            "industrial_area",
            "intersection",
            "island",
            "lake",
            "meadow",
            "medium_residential",
            "mobile_home_park",
            "mountain",
            "overpass",
            "palace",
            "parking_lot",
            "railway",
            "railway_station",
            "rectangular_farmland",
            "river",
            "roundabout",
            "runway",
            "sea_ice",
            "ship",
            "snowberg",
            "sparse_residential",
            "stadium",
            "storage_tank",
            "tennis_court",
            "terrace",
            "thermal_power_station",
            "wetland",
        ]
