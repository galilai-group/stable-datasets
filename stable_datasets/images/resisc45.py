import urllib.request
import zipfile
from pathlib import Path

import datasets

from stable_datasets.utils import BaseDatasetBuilder, _default_dest_folder


class RESISC45(BaseDatasetBuilder):
    """NWPU-RESISC45: Remote Sensing Image Scene Classification dataset with 45 classes.

    The `RESISC45 <http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html>`_ dataset
    (NWPU-RESISC45) was created by Northwestern Polytechnical University (NWPU).
    It is a publicly available benchmark for Remote Sensing Image Scene Classification (RESISC),
    created by Northwestern Polytechnical University (NWPU).

    The dataset contains 31,500 images, covering 45 scene classes with 700 images in each class.
    The images have a resolution of 256×256 pixels and were manually extracted from Google Earth
    imagery from over 100 countries and regions. Images have a pixel resolution of 0.2-30m per pixel
    and cover a wide range of remote sensing scene classes.
    """

    VERSION = datasets.Version("1.0.0")

    SOURCE = {
        "homepage": "http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html",
        "assets": {
            "train": "https://figshare.com/ndownloader/files/34054286",
        },
        "citation": """@article{cheng2017remote,
                         title={Remote sensing image scene classification: Benchmark and state of the art},
                         author={Cheng, Gong and Han, Junwei and Lu, Xiaoqiang},
                         journal={Proceedings of the IEEE},
                         volume={105},
                         number={10},
                         pages={1865--1883},
                         year={2017},
                         publisher={IEEE}}""",
    }

    def _info(self):
        return datasets.DatasetInfo(
            description="""NWPU-RESISC45 is a publicly available benchmark for Remote Sensing Image Scene Classification, created by Northwestern Polytechnical University. It contains 31,500 images covering 45 scene classes with 700 images in each class. Images are 256×256 RGB with pixel resolution ranging from 0.2-30m. The dataset covers diverse remote sensing scenes from over 100 countries. See http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html for more information.""",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(
                        names=[
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
                    ),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _split_generators(self, dl_manager):
        """Generate a single train split containing all dataset images."""
        # Use the standard download directory from BaseDatasetBuilder
        download_dir = getattr(self, "_raw_download_dir", None)
        if download_dir is None:
            download_dir = _default_dest_folder()
        download_dir = Path(download_dir)
        download_dir.mkdir(parents=True, exist_ok=True)

        # Download the dataset ZIP file from Figshare
        zip_url = "https://figshare.com/ndownloader/files/34054286"
        zip_path = download_dir / "NWPU-RESISC45.zip"

        if not zip_path.exists():
            print("Downloading RESISC45 dataset from Figshare...")
            urllib.request.urlretrieve(zip_url, zip_path)

        # Extract ZIP file if not already extracted
        extracted_path = download_dir / "NWPU"
        if not extracted_path.exists():
            print("Extracting ZIP archive...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(download_dir)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_path": str(extracted_path),
                },
            ),
        ]

    def _generate_examples(self, data_path):
        """Generate examples from all images in the dataset."""
        data_path = Path(data_path)

        idx = 0
        # Iterate through all class directories
        for class_dir in sorted(data_path.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            # Iterate through all images in each class directory
            for image_path in sorted(class_dir.glob("*.jpg")):
                yield idx, {"image": str(image_path), "label": class_name}
                idx += 1
