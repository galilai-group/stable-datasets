import os
import pickle
import tarfile
import time
from ..utils import Dataset
from io import BytesIO
import numpy as np
from tqdm import tqdm


class CIFAR10(Dataset):
    """Image classification.
    The `CIFAR-10 < https: // www.cs.toronto.edu/~kriz/cifar.html >`_ dataset
    was collected by Alex Krizhevsky, Vinod Nair, and Geoffrey
    Hinton. It consists of 60000 32x32 colour images in 10 classes, with
    6000 images per class. There are 50000 training images and 10000 test images.
    The dataset is divided into five training batches and one test batch,
    each with 10000 images. The test batch contains exactly 1000 randomly
    selected images from each class. The training batches contain the
    remaining images in random order, but some training batches may
    contain more images from one class than another. Between them, the
    training batches contain exactly 5000 images from each class.

    Parameters
    ----------

    path: str (optional)
        default ($DATASET_PATH), the path to look for the data and
        where the data will be downloaded if not present

    Returns
    -------

    train_images: array

    train_labels: array

    test_images: array

    test_labels: array

    """

    @property
    def md5(self):
        return {"cifar-10-python.tar.gz": "c58f30108f718f92721af3b95e74349a"}

    @property
    def urls(self):
        return {
            "cifar-10-python.tar.gz": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        }

    @property
    def num_classes(self):
        return 10

    @property
    def label_to_name(self, label):
        return {
            0: "airplane",
            1: "automobile",
            2: "bird",
            3: "cat",
            4: "deer",
            5: "dog",
            6: "frog",
            7: "horse",
            8: "sheep",
            9: "truck",
        }[label]

    def load(self):
        t0 = time.time()

        tar = tarfile.open(self.path / self.name / "cifar-10-python.tar.gz", "r:gz")

        # Load train set
        train_images = list()
        train_labels = list()
        for k in tqdm(range(1, 6), desc="Loading cifar10", ascii=True):
            f = tar.extractfile("cifar-10-batches-py/data_batch_" + str(k)).read()
            data_dic = pickle.loads(f, encoding="latin1")
            train_images.append(data_dic["data"].reshape((-1, 3, 32, 32)))
            train_labels.append(data_dic["labels"])
        train_images = np.concatenate(train_images, 0)
        train_labels = np.concatenate(train_labels, 0)

        # Load test set
        f = tar.extractfile("cifar-10-batches-py/test_batch").read()
        data_dic = pickle.loads(f, encoding="latin1")
        test_images = data_dic["data"].reshape((-1, 3, 32, 32))
        test_labels = np.array(data_dic["labels"])

        self["train_X"] = np.transpose(train_images, (0, 2, 3, 1))
        self["train_y"] = train_labels
        self["test_X"] = np.transpose(test_images, (0, 2, 3, 1))
        self["test_y"] = test_labels
        print("Dataset cifar10 loaded in{0:.2f}s.".format(time.time() - t0))
        return self


class CIFAR10C(CIFAR10):
    """Image classification.
    The `CIFAR-10 < https: // www.cs.toronto.edu/~kriz/cifar.html >`_ dataset
    was collected by Alex Krizhevsky, Vinod Nair, and Geoffrey
    Hinton. It consists of 60000 32x32 colour images in 10 classes, with
    6000 images per class. There are 50000 training images and 10000 test images.
    The dataset is divided into five training batches and one test batch,
    each with 10000 images. The test batch contains exactly 1000 randomly
    selected images from each class. The training batches contain the
    remaining images in random order, but some training batches may
    contain more images from one class than another. Between them, the
    training batches contain exactly 5000 images from each class.

    Parameters
    ----------

    path: str
        default ($DATASET_PATH), the path to look for the data and
        where the data will be downloaded if not present

    corruption: str or list
        which corruption version to use

    Keys
    -------

    X
    y
    corruption_name
    corruption_level
    """

    @property
    def corruptions(self):
        return [
            "zoom_blur",
            "speckle_noise",
            "spatter",
            "snow",
            "shot_noise",
            "saturate",
            "pixelate",
            "motion_blur",
            "jpeg_compression",
            "impulse_noise",
            "glass_blur",
            "gaussian_noise",
            "gaussian_blur",
            "frost",
            "fog",
            "elastic_transform",
            "defocus_blur",
            "contrast",
            "bightness",
        ]

    @property
    def urls(self):
        return {
            "CIFAR-10-C.tar": "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1"
        }

    @property
    def md5(self):
        return {"CIFAR-10-C.tar": "56bf5dcef84df0e2308c6dcbcbbd8499"}

    @property
    def webpage(self):
        return "https://zenodo.org/records/2535967"

    def load(self, corruption=None):
        t0 = time.time()

        tar = tarfile.open(self.path / self.name / "CIFAR-10-C.tar", "r")

        # Load train set
        array_file = BytesIO()
        array_file.write(tar.extractfile("CIFAR-10-C/labels.npy").read())
        array_file.seek(0)
        labels = np.load(array_file)
        if type(corruption) == str:
            corruptions = [corruption]
        elif type(corruption) in [list, tuple]:
            corruptions = corruption
        else:
            corruptions = self.corruptions
        images = []
        names = []
        for c in corruptions:
            assert c in self.corruptions
            print(f"Loading corruption {c}")
            array_file = BytesIO()
            array_file.write(tar.extractfile(f"CIFAR-10-C/{c}.npy").read())
            array_file.seek(0)
            images.append(np.load(array_file))
            names.extend([c] * 10000)

        self["X"] = np.concatenate(images).transpose((0, 2, 3, 1))
        self["y"] = np.concatenate([labels for i in range(len(corruptions))])
        self["corruption_name"] = np.array(names)
        self["corruption_level"] = (
            np.arange(1, 6).repeat(10000).repeat(len(corruptions))
        )
        print("Dataset cifar10-C loaded in{0:.2f}s.".format(time.time() - t0))
        return self
