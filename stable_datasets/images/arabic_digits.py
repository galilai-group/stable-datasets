import io
from tqdm import tqdm
from PIL import Image
from zipfile import ZipFile
import datasets


class ArabicDigits(datasets.GeneratorBasedBuilder):
    """Arabic Handwritten Digits Dataset."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="""Arabic Handwritten Digits Dataset, composed of images of Arabic digits handwritten 
                           by participants. This dataset is structured for use in machine learning tasks such 
                           as digit classification.""",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(names=[str(i) for i in range(10)])
                }
            ),
            supervised_keys=("image", "label"),
            homepage="https://github.com/mloey/Arabic-Handwritten-Digits-Dataset",
            citation="""@inproceedings{el2016cnn,
                        title={CNN for handwritten arabic digits recognition based on LeNet-5},
                        author={El-Sawy, Ahmed and Hazem, EL-Bakry and Loey, Mohamed},
                        booktitle={International conference on advanced intelligent systems and informatics},
                        pages={566--575},
                        year={2016},
                        organization={Springer}
                        }"""
        )

    def _split_generators(self, dl_manager):
        urls = {
            "train": "https://github.com/mloey/Arabic-Handwritten-Digits-Dataset/raw/master/Train%20Images.zip",
            "test": "https://github.com/mloey/Arabic-Handwritten-Digits-Dataset/raw/master/Test%20Images.zip"
        }
        downloaded_files = dl_manager.download(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"archive_path": downloaded_files["train"], "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"archive_path": downloaded_files["test"], "split": "test"},
            )
        ]

    def _generate_examples(self, archive_path, split):
        """Generate examples from the ZIP archives of images and labels."""
        with ZipFile(archive_path, "r") as archive:
            for entry in tqdm(archive.infolist(), desc=f"Processing {split} set"):
                if entry.filename.endswith(".png"):
                    content = archive.read(entry)
                    image = Image.open(io.BytesIO(content))
                    label = int(entry.filename.split("_")[-1][:-4])  # Extract label from filename

                    yield entry.filename, {
                        "image": image,
                        "label": label
                    }
