import tarfile

import datasets
from PIL import Image

from stable_datasets.utils import BaseDatasetBuilder


class FGVCAircraft(BaseDatasetBuilder):
    """Fine-Grained Visual Classification of Aircraft (FGVC-Aircraft) Dataset.

    FGVC-Aircraft is a benchmark dataset for fine-grained visual categorization of aircraft.
    The dataset contains 10,000 images of aircraft with 100 different aircraft model variants.
    Aircraft models are organized in a hierarchical structure with three levels: variant (finest),
    family, and manufacturer (coarsest).

    The dataset is divided into training (3,334 images), validation (3,333 images), and test
    (3,333 images) subsets. Images are about 1-2MP resolution with a 20-pixel copyright banner
    at the bottom that is automatically removed during loading.

    Usage:
        dataset = FGVCAircraft(config_name="variant", split="train")
        dataset = FGVCAircraft(config_name="family", split="train")
        dataset = FGVCAircraft(config_name="manufacturer", split="train")
    """

    VERSION = datasets.Version("1.0.0")

    SOURCE = {
        "homepage": "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/",
        "assets": {
            "train": "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz",
            "validation": "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz",
            "test": "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz",
        },
        "citation": """@techreport{maji13fine-grained,
                        title         = {Fine-Grained Visual Classification of Aircraft},
                        author        = {S. Maji and J. Kannala and E. Rahtu and M. Blaschko and A. Vedaldi},
                        year          = {2013},
                        archivePrefix = {arXiv},
                        eprint        = {1306.5151},
                        primaryClass  = "cs.CV",
                    }""",
        "license": "Unknown",
    }

    def __init__(self, config_name="variant", **kwargs):
        """
        Initialize FGVC Aircraft dataset.

        Args:
            config_name: str, one of "variant", "family", or "manufacturer"
                - variant: 100 fine-grained aircraft model variants (default)
                - family: 70 aircraft families
                - manufacturer: 30 aircraft manufacturers
            **kwargs: Additional arguments passed to BaseDatasetBuilder
        """
        if config_name not in ["variant", "family", "manufacturer"]:
            raise ValueError(f"config_name must be one of 'variant', 'family', or 'manufacturer', got '{config_name}'")
        self.config_name = config_name
        super().__init__(**kwargs)

    def _info(self):
        # Determine class labels based on config name
        if self.config_name == "variant":
            class_names = self._variant_labels()
            description = "100 aircraft model variants (finest granularity)"
        elif self.config_name == "family":
            class_names = self._family_labels()
            description = "70 aircraft families (medium granularity)"
        elif self.config_name == "manufacturer":
            class_names = self._manufacturer_labels()
            description = "30 aircraft manufacturers (coarsest granularity)"

        return datasets.DatasetInfo(
            description=f"""Fine-Grained Visual Classification of Aircraft (FGVC-Aircraft) dataset.
                           Classification granularity: {description}.
                           The dataset contains 10,000 images of aircraft organized in three splits
                           (train: 3,334, val: 3,333, test: 3,333). Images have a 20-pixel copyright
                           banner at the bottom that is automatically removed.""",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(names=class_names),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license=self.SOURCE["license"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        """Generate examples from the tar.gz archive."""
        # Map validation to val for internal tar.gz file structure
        internal_split = "val" if split == "validation" else split

        with tarfile.open(data_path, "r:gz") as tar:
            # Read only the label file we need based on config_name
            label_file = f"fgvc-aircraft-2013b/data/images_{self.config_name}_{internal_split}.txt"

            # Load annotations
            label_dict = {}
            with tar.extractfile(label_file) as f:
                for line in f:
                    line = line.decode("utf-8").strip()
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        image_id, label = parts
                        label_dict[image_id] = label

            # Iterate through tar members once (much faster than repeated getmember calls)
            for member in tar.getmembers():
                if not member.isfile():
                    continue

                # Check if this is an image file we need
                if member.name.startswith("fgvc-aircraft-2013b/data/images/") and member.name.endswith(".jpg"):
                    # Extract image_id from path: "fgvc-aircraft-2013b/data/images/0787226.jpg" -> "0787226"
                    image_id = member.name.split("/")[-1].replace(".jpg", "")

                    if image_id in label_dict:
                        try:
                            image_file = tar.extractfile(member)
                            image = Image.open(image_file)

                            # Remove the bottom 20 pixels copyright banner
                            cropped_image = image.crop((0, 0, image.width, image.height - 20))

                            yield (
                                image_id,
                                {
                                    "image": cropped_image,
                                    "label": label_dict[image_id],
                                },
                            )
                        except Exception:
                            # Skip if image cannot be processed
                            continue

    @staticmethod
    def _variant_labels():
        """Returns the list of 100 aircraft model variants."""
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

    @staticmethod
    def _family_labels():
        """Returns the list of 70 aircraft families."""
        return [
            "A300",
            "A310",
            "A320",
            "A330",
            "A340",
            "A380",
            "ATR-42",
            "ATR-72",
            "An-12",
            "BAE 146",
            "BAE-125",
            "Beechcraft 1900",
            "Boeing 707",
            "Boeing 717",
            "Boeing 727",
            "Boeing 737",
            "Boeing 747",
            "Boeing 757",
            "Boeing 767",
            "Boeing 777",
            "C-130",
            "C-47",
            "CRJ-200",
            "CRJ-700",
            "Cessna 172",
            "Cessna 208",
            "Cessna Citation",
            "Challenger 600",
            "DC-10",
            "DC-3",
            "DC-6",
            "DC-8",
            "DC-9",
            "DH-82",
            "DHC-1",
            "DHC-6",
            "DR-400",
            "Dash 8",
            "Dornier 328",
            "EMB-120",
            "Embraer E-Jet",
            "Embraer ERJ 145",
            "Embraer Legacy 600",
            "Eurofighter Typhoon",
            "F-16",
            "F/A-18",
            "Falcon 2000",
            "Falcon 900",
            "Fokker 100",
            "Fokker 50",
            "Fokker 70",
            "Global Express",
            "Gulfstream",
            "Hawk T1",
            "Il-76",
            "King Air",
            "L-1011",
            "MD-11",
            "MD-80",
            "MD-90",
            "Metroliner",
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

    @staticmethod
    def _manufacturer_labels():
        """Returns the list of 30 aircraft manufacturers."""
        return [
            "ATR",
            "Airbus",
            "Antonov",
            "Beechcraft",
            "Boeing",
            "Bombardier Aerospace",
            "British Aerospace",
            "Canadair",
            "Cessna",
            "Cirrus Aircraft",
            "Dassault Aviation",
            "Dornier",
            "Douglas Aircraft Company",
            "Embraer",
            "Eurofighter",
            "Fairchild",
            "Fokker",
            "Gulfstream Aerospace",
            "Ilyushin",
            "Lockheed Corporation",
            "Lockheed Martin",
            "McDonnell Douglas",
            "Panavia",
            "Piper",
            "Robin",
            "Saab",
            "Supermarine",
            "Tupolev",
            "Yakovlev",
            "de Havilland",
        ]
