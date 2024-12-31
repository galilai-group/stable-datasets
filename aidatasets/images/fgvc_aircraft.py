import datasets
from PIL import Image
import os


class FGVCAircraft(datasets.GeneratorBasedBuilder):
    """FGVC Aircraft Dataset."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="The FGVC Aircraft dataset for fine-grained visual categorization.",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(names=self._labels())
                }
            ),
            supervised_keys=("image", "label"),
            homepage="https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/",
            citation="""@article{maji2013fgvc,
                         title={Fine-Grained Visual Classification of Aircraft},
                         author={Maji, Subhransu and Rahtu, Esa and Kannala, Juho and Blaschko, Matthew and Vedaldi, Andrea},
                         journal={arXiv preprint arXiv:1306.5151},
                         year={2013}}"""
        )

    def _split_generators(self, dl_manager):
        archive_path = dl_manager.download_and_extract(
            "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
        )
        base_path = os.path.join(archive_path, "fgvc-aircraft-2013b", "data")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"base_dir": base_path, "split_file": "images_variant_train.txt"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"base_dir": base_path, "split_file": "images_variant_test.txt"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"base_dir": base_path, "split_file": "images_variant_val.txt"}
            )
        ]

    def _generate_examples(self, base_dir, split_file):
        with open(os.path.join(base_dir, split_file), "r") as f:
            for idx, line in enumerate(f):
                parts = line.strip().split(maxsplit=1)
                image_id = parts[0]
                label = parts[1] if len(parts) > 1 else None
                image_path = os.path.join(base_dir, 'images', f"{image_id}.jpg")
                if os.path.exists(image_path):
                    # Remove the bottom 20 pixels from the image to remove the copyright banner
                    image = Image.open(image_path)
                    cropped_image = image.crop((0, 0, image.width, image.height - 20))
                    yield idx, {
                        "image": cropped_image,
                        "label": label,
                    }

    @staticmethod
    def _labels():
        return [
            "707-320", "727-200", "737-200", "737-300", "737-400", "737-500", "737-600",
            "737-700", "737-800", "737-900", "747-100", "747-200", "747-300", "747-400",
            "757-200", "757-300", "767-200", "767-300", "767-400", "777-200", "777-300",
            "A300B4", "A310", "A318", "A319", "A320", "A321", "A330-200", "A330-300",
            "A340-200", "A340-300", "A340-500", "A340-600", "A380", "ATR-42", "ATR-72",
            "An-12", "BAE 146-200", "BAE 146-300", "BAE-125", "Beechcraft 1900",
            "Boeing 717", "C-130", "C-47", "CRJ-200", "CRJ-700", "CRJ-900", "Cessna 172",
            "Cessna 208", "Cessna 525", "Cessna 560", "Challenger 600", "DC-10", "DC-3",
            "DC-6", "DC-8", "DC-9-30", "DH-82", "DHC-1", "DHC-6", "DHC-8-100",
            "DHC-8-300", "DR-400", "Dornier 328", "E-170", "E-190", "E-195", "EMB-120",
            "ERJ 135", "ERJ 145", "Embraer Legacy 600", "Eurofighter Typhoon",
            "F-16A/B", "F/A-18", "Falcon 2000", "Falcon 900", "Fokker 100", "Fokker 50",
            "Fokker 70", "Global Express", "Gulfstream IV", "Gulfstream V", "Hawk T1",
            "Il-76", "L-1011", "MD-11", "MD-80", "MD-87", "MD-90", "Metroliner",
            "Model B200", "PA-28", "SR-20", "Saab 2000", "Saab 340", "Spitfire",
            "Tornado", "Tu-134", "Tu-154", "Yak-42"
        ]
