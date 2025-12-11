import rarfile
import datasets
from io import BytesIO


class Linnaeus5(datasets.GeneratorBasedBuilder):
    """Linnaeus 5 Dataset: RGB images (256x256) for classification across 5 categories."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="""Linnaeus 5 dataset contains RGB images (256x256) for classification across 5 categories: 
                           berry, bird, dog, flower, and other (negative set). It includes 1200 training images 
                           and 400 test images per class.""",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(names=["berry", "bird", "dog", "flower", "other"]),
                }
            ),
            supervised_keys=("image", "label"),
            homepage="http://chaladze.com/l5/",
            citation="""@article{chaladze2017linnaeus,
                      title={Linnaeus 5 dataset for machine learning},
                      author={Chaladze, G and Kalatozishvili, L},
                      journal={chaladze. com},
                      year={2017}}
                      """
        )

    def _split_generators(self, dl_manager):
        archive_path = dl_manager.download(
            "http://chaladze.com/l5/img/Linnaeus%205%20256X256.rar"
        )
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"archive_path": archive_path, "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"archive_path": archive_path, "split": "test"},
            ),
        ]

    def _generate_examples(self, archive_path, split):
        with rarfile.RarFile(archive_path) as rar:
            for member in rar.infolist():
                if split in member.filename and member.filename.endswith(".jpg"):
                    label = member.filename.split("/")[2]
                    with rar.open(member) as file:
                        image_bytes = BytesIO(file.read())
                        yield member.filename, {
                            "image": image_bytes,
                            "label": label,
                        }
