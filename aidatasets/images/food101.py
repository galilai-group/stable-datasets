import os
import pickle
import tarfile
import time
from ..utils import Dataset
from PIL import Image

import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


class Food101(Dataset):
    """Image classification."""

    @property
    def urls(self):
        return {"food-101.tar.gz": "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"}

    @property
    def extract(self):
        return ["food-101.tar.gz"]

    @property
    def num_classes(self):
        return 101

    @property
    def label_to_name(self, label):
        return self.loaded_names[label]

    @property
    def name(self):
        return "Food101"

    @property
    def webpage(self):
        return "https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/"

    @property
    def cite(self):
        return """@inproceedings{bossard14,
title = {Food-101 -- Mining Discriminative Components with Random Forests},
author = {Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc},
booktitle = {European Conference on Computer Vision},
year = {2014}
}"""

    def load(self):
        t0 = time.time()

        tar = tarfile.open(self.path / self.name / list(self.urls.keys())[0], "r:gz")

        print("Loading labels")
        self["labels"] = (
            (self.path / self.name / "extracted_food-101.tar/food-101/meta/labels.txt")
            .read_text()
            .splitlines()
        )
        self["classes"] = (
            (self.path / self.name / "extracted_food-101.tar/food-101/meta/classes.txt")
            .read_text()
            .splitlines()
        )

        print("Loading train info")
        train = (
            (self.path / self.name / "extracted_food-101.tar/food-101/meta/train.txt")
            .read_text()
            .splitlines()
        )
        print("Loading test info")
        test = (
            (self.path / self.name / "extracted_food-101.tar/food-101/meta/test.txt")
            .read_text()
            .splitlines()
        )

        # Load train set
        train_images = list()
        train_labels = list()
        for name in tqdm(train, desc="Loading Train Food101", ascii=True):
            train_images.append(
                self.path
                / self.name
                / f"extracted_food-101.tar/food-101/images/{name}.jpg"
            )
            train_labels.append(name.split("/")[0])

        label_encoder = LabelEncoder().fit(train_labels)
        train_labels = label_encoder.transform(train_labels)

        # Load test set
        test_images = list()
        test_labels = list()
        for name in tqdm(test, desc="Loading Test Food101", ascii=True):
            test_images.append(
                self.path
                / self.name
                / f"extracted_food-101.tar/food-101/images/{name}.jpg"
            )
            test_labels.append(name.split("/")[0])

        test_labels = label_encoder.transform(test_labels)
        self["train_X"] = ImagePathsDataset(train_images)
        self["train_y"] = train_labels
        self["test_X"] = ImagePathsDataset(test_images)
        self["test_y"] = test_labels
        print("Dataset Food101 loaded in{0:.2f}s.".format(time.time() - t0))
