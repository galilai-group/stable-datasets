import datasets
from PIL import Image
import zipfile


class Beans(datasets.GeneratorBasedBuilder):
    """Bean disease dataset for classification of three classes: Angular Leaf Spot, Bean Rust, and Healthy leaves."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="""The IBeans dataset contains leaf images representing three classes: 
                1) Healthy leaves, 2) Angular Leaf Spot, and 3) Bean Rust. Images are collected in Uganda for disease 
                classification in the field.""",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(
                        names=["healthy", "angular_leaf_spot", "bean_rust"]
                    ),
                }
            ),
            supervised_keys=("image", "label"),
            license="MIT License",
            citation="""@misc{makerere2020beans,
                         author = "{Makerere AI Lab}",
                         title = "{Bean Disease Dataset}",
                         year = "2020",
                         month = "January",
                         url = "https://github.com/AI-Lab-Makerere/ibean/"}"""
        )

    def _split_generators(self, dl_manager):
        urls = {
            "train": "https://storage.googleapis.com/ibeans/train.zip",
            "test": "https://storage.googleapis.com/ibeans/test.zip",
            "validation": "https://storage.googleapis.com/ibeans/validation.zip",
        }
        downloaded_files = dl_manager.download(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"zip_path": downloaded_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"zip_path": downloaded_files["test"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"zip_path": downloaded_files["validation"]},
            ),
        ]

    def _generate_examples(self, zip_path):
        with zipfile.ZipFile(zip_path, "r") as archive:
            for file_name in archive.namelist():
                if file_name.endswith(".jpg"):
                    with archive.open(file_name) as file:
                        image_data = Image.open(file)
                        label_name = file_name.split("/")[1]
                        label = self.info.features["label"].str2int(label_name)
                        yield file_name, {"image": image_data, "label": label}
