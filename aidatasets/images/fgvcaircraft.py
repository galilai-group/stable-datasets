import os
import pickle
import tarfile
import time
from ..utils import Dataset
from io import BytesIO
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


class FGVCAircraft(Dataset):
    """Image classification."""

    @property
    def urls(self):
        return {
            "fgvc-aircraft-2013b.tar.gz": "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
        }

    @property
    def extract(self):
        return ["fgvc-aircraft-2013b.tar.gz"]

    @property
    def num_classes(self):
        return 102

    @property
    def webpage(self):
        return "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/"

    def load(self):
        t0 = time.time()
        base = (
            self.path
            / self.name
            / "extracted_fgvc-aircraft-2013b.tar/fgvc-aircraft-2013b/data"
        )
        for n, m in zip(
            ["variant", "manufacturer", "family"],
            ["variants", "manufacturers", "families"],
        ):
            self[n] = (base / (m + ".txt")).read_text().splitlines()
            for p in ["train", "test", "val"]:
                self[p + "_" + n] = (
                    (base / f"images_{n}_{p}.txt").read_text().splitlines()
                )
                self[p + "_" + n] = [
                    self[n].index(" ".join(i.split(" ")[1:])) for i in self[p + "_" + n]
                ]

        val_names = (base / "images_val.txt").read_text().splitlines()
        train_names = (base / "images_train.txt").read_text().splitlines()
        test_names = (base / "images_test.txt").read_text().splitlines()

        train_images, test_images, val_images = [], [], []
        for images, names, desc in zip(
            [train_images, test_images, val_images],
            [train_names, test_names, val_names],
            ["Train", "Test", "Val"],
        ):
            for name in tqdm(names, desc=f"{desc} images"):
                images.append(base / "images" / f"{name}.jpg")
        self["train_X"] = ImagePathsDataset(train_images)
        self["test_X"] = ImagePathsDataset(test_images)
        self["val_X"] = ImagePathsDataset(val_images)
        self["train_y"] = self["train_variant"]
        self["test_y"] = self["test_variant"]
        self["val_y"] = self["val_variant"]
        print(f"Dataset {self.name} loaded in {0:.2f}s.".format(time.time() - t0))
