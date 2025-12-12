#!/usr/bin/env python

"""Legacy Cassava loader (to be refactored into a BaseDatasetBuilder).

This module was moved under `stable_datasets.images` to align the repository layout.
It still exposes the original imperative `cassava.load(...)` API for now.
"""

import io
import os
import time
import urllib
import zipfile

import matplotlib.image as mpimg
import numpy as np


__author__ = "Randall Balestriero"


class cassava:
    """Plant images classification.

    The data consists of two folders, a training folder that contains 5
    subfolders that contain the respective images for the different 5 classes
    and a test folder containing test images.
    """

    classes = ["cbb", "cmd", "cbsd", "cgm", "healthy"]

    @staticmethod
    def download(path):
        # Check if directory exists
        if not os.path.isdir(path + "cassava"):
            print("Creating cassava Directory")
            os.mkdir(path + "cassava")
        # Check if file exists
        if not os.path.exists(path + "cassava/cassavaleafdata.zip"):
            url = "https://storage.googleapis.com/emcassavadata/" + "cassavaleafdata.zip"
            urllib.request.urlretrieve(url, path + "cassava/cassavaleafdata.zip")

    @staticmethod
    def load(path=None):
        if path is None:
            path = os.environ["DATASET_PATH"]

        cassava.download(path)

        t0 = time.time()

        # Loading the file
        data = {"train": [[], []], "test": [[], []], "validation": [[], []]}

        f = zipfile.ZipFile(path + "cassava/cassavaleafdata.zip")
        for filename in f.namelist():
            if ".jpg" not in filename:
                continue
            setname, foldername = filename.split("/")[1:3]
            img = mpimg.imread(io.BytesIO(f.read(filename)), "jpg")
            data[setname][0].append(img)
            data[setname][1].append(cassava.classes.index(foldername))

        train_images = np.array(data["train"][0])
        test_images = np.array(data["test"][0])
        valid_images = np.array(data["validation"][0])

        train_labels = np.array(data["train"][1])
        test_labels = np.array(data["test"][1])
        valid_labels = np.array(data["validation"][1])

        print(f"Dataset cassava loaded in {time.time() - t0:.2f}s.")

        return (
            train_images,
            train_labels,
            valid_images,
            valid_labels,
            test_images,
            test_labels,
        )
