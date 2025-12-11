import datasets
import os
from PIL import Image


class RockPaperScissor(datasets.GeneratorBasedBuilder):
    """Rock Paper Scissors dataset."""
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="Rock Paper Scissors contains images from various hands, from different races, ages, and "
                        "genders, posed into Rock / Paper or Scissors and labeled as such.",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(names=["rock", "paper", "scissors"]),
                }
            ),
            supervised_keys=("image", "label"),
            homepage="https://laurencemoroney.com/datasets.html",
            license="CC By 2.0",
        )

    def _split_generators(self, dl_manager):
        urls = {
            "train": "https://storage.googleapis.com/download.tensorflow.org/data/rps.zip",
            "test": "https://storage.googleapis.com/download.tensorflow.org/data/rps-test-set.zip",
        }
        extracted_paths = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_dir": extracted_paths["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"data_dir": extracted_paths["test"]},
            ),
        ]

    def _generate_examples(self, data_dir):
        for root, _, files in os.walk(data_dir):
            for file_name in files:
                if file_name.endswith(".png"):
                    label = os.path.basename(root)  # Folder name as label
                    file_path = os.path.join(root, file_name)
                    # Open image and ensure it is RGB
                    with open(file_path, "rb") as img_file:
                        image = Image.open(img_file).convert("RGB")
                        yield file_path, {"image": image, "label": label}
