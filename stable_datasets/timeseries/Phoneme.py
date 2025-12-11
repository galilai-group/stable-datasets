import os
import numpy as np
from scipy.io import arff
from ..utils import download_dataset
import pathlib


def load(path=None):
    """See http://www.timeseriesclassification.com/description.php?Dataset=Phoneme
    Parameters
    ----------
        path: str (optional)
            default ($DATASET_PATH), the path to look for the data and
            where the data will be downloaded if not present

    Returns
    -------

        train_images: array

        train_labels: array

        valid_images: array

        valid_labels: array

        test_images: array

        test_labels: array

    """

    if path is None:
        path = os.environ["DATASET_PATH"]

    path = pathlib.Path(path) / "Phoneme"
    download_dataset(
        path,
        {
            "Phoneme.zip": "http://www.timeseriesclassification.com/Downloads/Phoneme.zip"
        },
        extract=True,
    )

    path = path / "extracted_Phoneme"

    data_train = arff.loadarff(path / "Phoneme_TRAIN.arff")
    data_test = arff.loadarff(path / "Phoneme_TEST.arff")

    data_train = np.asarray([data_train[0][name] for name in data_train[1].names()])
    X_train = data_train[:-1].T.astype("float64")
    y_train = data_train[-1]

    data_test = np.asarray([data_test[0][name] for name in data_test[1].names()])
    X_test = data_test[:-1].T.astype("float64")
    y_test = data_test[-1]
    return (X_train, y_train), (X_test, y_test)
