import os
import tarfile
from pathlib import Path

from PIL import Image as PILImage

from stable_datasets.schema import ClassLabel, DatasetInfo, Features, Image as ImageFeature, Version
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, download


class FGVCAircraft(BaseDatasetBuilder):
    """Fine-Grained Visual Classification of Aircraft (FGVC-Aircraft) Dataset."""

    VERSION = Version("1.0.0")

    SOURCE = {
        "homepage": "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/",
        "citation": """@article{maji2013fgvc,
                         title={Fine-Grained Visual Classification of Aircraft},
                         author={Maji, Subhransu and Rahtu, Esa and Kannala, Juho and Blaschko, Matthew and Vedaldi, Andrea},
                         journal={arXiv preprint arXiv:1306.5151},
                         year={2013}}""",
        "assets": {
            "archive": "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz",
        },
    }

    def _info(self):
        return DatasetInfo(
            description="The FGVC Aircraft dataset for fine-grained visual categorization.",
            features=Features(
                {"image": ImageFeature(), "label": ClassLabel(names=self._labels())}
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _split_generators(self, dl_manager=None):
        source = self._source()
        archive_url = source["assets"]["archive"]
        archive_path = download(archive_url, dest_folder=self._raw_download_dir)

        # Extract the tarball
        extract_dir = Path(self._raw_download_dir) / "fgvc-aircraft-extracted"
        if not extract_dir.exists():
            extract_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(extract_dir)

        base_path = os.path.join(extract_dir, "fgvc-aircraft-2013b", "data")
        return [
            SplitGenerator(
                name=Split.TRAIN, gen_kwargs={"base_dir": base_path, "split_file": "images_variant_train.txt"}
            ),
            SplitGenerator(
                name=Split.TEST, gen_kwargs={"base_dir": base_path, "split_file": "images_variant_test.txt"}
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={"base_dir": base_path, "split_file": "images_variant_val.txt"},
            ),
        ]

    def _generate_examples(self, base_dir, split_file):
        with open(os.path.join(base_dir, split_file)) as f:
            for idx, line in enumerate(f):
                parts = line.strip().split(maxsplit=1)
                image_id = parts[0]
                label = parts[1] if len(parts) > 1 else None
                image_path = os.path.join(base_dir, "images", f"{image_id}.jpg")
                if os.path.exists(image_path):
                    with PILImage.open(image_path) as image:
                        cropped_image = image.crop((0, 0, image.width, image.height - 20))
                        yield (
                            idx,
                            {
                                "image": cropped_image,
                                "label": label,
                            },
                        )

    @staticmethod
    def _labels():
        return [
            "707-320",
            "727-200",
            "737-200",
            "737-300",
            "737-400",
            "737-500",
            "737-600",
            "737-700",
            "737-800",
            "737-900",
            "747-100",
            "747-200",
            "747-300",
            "747-400",
            "757-200",
            "757-300",
            "767-200",
            "767-300",
            "767-400",
            "777-200",
            "777-300",
            "A300B4",
            "A310",
            "A318",
            "A319",
            "A320",
            "A321",
            "A330-200",
            "A330-300",
            "A340-200",
            "A340-300",
            "A340-500",
            "A340-600",
            "A380",
            "ATR-42",
            "ATR-72",
            "An-12",
            "BAE 146-200",
            "BAE 146-300",
            "BAE-125",
            "Beechcraft 1900",
            "Boeing 717",
            "C-130",
            "C-47",
            "CRJ-200",
            "CRJ-700",
            "CRJ-900",
            "Cessna 172",
            "Cessna 208",
            "Cessna 525",
            "Cessna 560",
            "Challenger 600",
            "DC-10",
            "DC-3",
            "DC-6",
            "DC-8",
            "DC-9-30",
            "DH-82",
            "DHC-1",
            "DHC-6",
            "DHC-8-100",
            "DHC-8-300",
            "DR-400",
            "Dornier 328",
            "E-170",
            "E-190",
            "E-195",
            "EMB-120",
            "ERJ 135",
            "ERJ 145",
            "Embraer Legacy 600",
            "Eurofighter Typhoon",
            "F-16A/B",
            "F/A-18",
            "Falcon 2000",
            "Falcon 900",
            "Fokker 100",
            "Fokker 50",
            "Fokker 70",
            "Global Express",
            "Gulfstream IV",
            "Gulfstream V",
            "Hawk T1",
            "Il-76",
            "L-1011",
            "MD-11",
            "MD-80",
            "MD-87",
            "MD-90",
            "Metroliner",
            "Model B200",
            "PA-28",
            "SR-20",
            "Saab 2000",
            "Saab 340",
            "Spitfire",
            "Tornado",
            "Tu-134",
            "Tu-154",
            "Yak-42",
        ]
