import zipfile
from pathlib import PurePosixPath

from PIL import Image as PILImage
from tqdm import tqdm

from stable_datasets.schema import ClassLabel, DatasetInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


class AID(BaseDatasetBuilder):
    """AID: 10,000 native-resolution aerial images in 30 scene categories.

    The complete collection is exposed as ``train``. The original benchmark
    evaluates repeated random partitions, not a fixed official holdout split.
    """

    VERSION = Version("1.0.0")

    SOURCE = {
        "homepage": "https://captain-whu.github.io/AID/",
        "assets": {
            "train": (
                "https://huggingface.co/datasets/Vision-Flan/vision-flan/resolve/"
                "a7fc1f3307439d00e39acb9a944356b346531ae9/shard_05/raw_dataset/AID.zip"
            ),
        },
        "mirror": "https://huggingface.co/datasets/Vision-Flan/vision-flan",
        "citation": """@article{xia2017aid,
            title={AID: A Benchmark Data Set for Performance Evaluation of Aerial Scene Classification},
            author={Xia, Gui-Song and Hu, Jingwen and Hu, Fan and Shi, Baoguang and Bai, Xiang and Zhong, Yanfei and Zhang, Liangpei and Lu, Xiaoqiang},
            journal={IEEE Transactions on Geoscience and Remote Sensing},
            volume={55},
            number={7},
            pages={3965--3981},
            year={2017},
            doi={10.1109/TGRS.2017.2685945}
        }""",
    }

    def _info(self):
        return DatasetInfo(
            description=(
                "AID: 10,000 aerial scene images in 30 classes at their native 600x600 resolution. "
                "The full collection is exposed as train; no official fixed holdout is provided."
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
                # The pinned raw-data mirror wraps the original AID directory once.
                if len(path.parts) != 4 or path.parts[:2] != ("AID", "AID"):
                    continue
                class_name = path.parent.name
                if class_name not in class_to_label:
                    raise ValueError(f"Unknown AID class {class_name!r} in {entry.filename!r}.")
                with archive.open(entry) as image_file:
                    with PILImage.open(image_file) as source_image:
                        image = source_image.convert("RGB")
                found_image = True
                yield entry.filename, {"image": image, "label": class_to_label[class_name]}
        if not found_image:
            raise ValueError("No AID images found under AID/AID/<class>/ in the archive.")

    @staticmethod
    def _labels():
        """Original class-directory names, in fixed alphabetical order."""
        return [
            "Airport",
            "BareLand",
            "BaseballField",
            "Beach",
            "Bridge",
            "Center",
            "Church",
            "Commercial",
            "DenseResidential",
            "Desert",
            "Farmland",
            "Forest",
            "Industrial",
            "Meadow",
            "MediumResidential",
            "Mountain",
            "Park",
            "Parking",
            "Playground",
            "Pond",
            "Port",
            "RailwayStation",
            "Resort",
            "River",
            "School",
            "SparseResidential",
            "Square",
            "Stadium",
            "StorageTanks",
            "Viaduct",
        ]
