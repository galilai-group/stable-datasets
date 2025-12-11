import os
import tarfile
import scipy.io
from PIL import Image
import numpy as np
import datasets


class Flowers102(datasets.GeneratorBasedBuilder):
    """Flowers102 Dataset."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="""The Flowers102 dataset is an image classification dataset consisting of 102 flower categories commonly found in the UK. Each category contains between 40 and 258 images.""",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(num_classes=102),
                }
            ),
            supervised_keys=("image", "label"),
            homepage="https://www.robots.ox.ac.uk/~vgg/data/flowers/102/",
            citation="""@inproceedings{nilsback2008flowers102,
                         title={Automated flower classification over a large number of classes},
                         author={Nilsback, Maria-Elena and Zisserman, Andrew},
                         booktitle={2008 Sixth Indian conference on computer vision, graphics \& image processing},
                         pages={722--729},
                         year={2008},
                         organization={IEEE}}""",
        )

    def _split_generators(self, dl_manager):
        urls = {
            "images": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz",
            "labels": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat",
            "setid": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat",
        }
        downloaded_files = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "image_dir": downloaded_files["images"],
                    "labels_path": downloaded_files["labels"],
                    "setid_path": downloaded_files["setid"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "image_dir": downloaded_files["images"],
                    "labels_path": downloaded_files["labels"],
                    "setid_path": downloaded_files["setid"],
                    "split": "valid",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "image_dir": downloaded_files["images"],
                    "labels_path": downloaded_files["labels"],
                    "setid_path": downloaded_files["setid"],
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, image_dir, labels_path, setid_path, split):
        labels = scipy.io.loadmat(labels_path)["labels"][0] - 1
        setid = scipy.io.loadmat(setid_path)
        if split == "train":
            ids = setid["trnid"][0]
        elif split == "valid":
            ids = setid["valid"][0]
        else:  # test
            ids = setid["tstid"][0]

        for idx, image_id in enumerate(ids):
            image_path = os.path.join(image_dir, "jpg", f"image_{image_id:05d}.jpg")
            yield idx, {
                "image": Image.open(image_path).convert("RGB"),
                "label": labels[image_id - 1],
            }
