import os
import numpy as np
from scipy.io import arff
from tqdm import tqdm
from ..utils import download_dataset
import pathlib
from multiprocessing import Pool


def load(path=None, num_workers=16):
    """See http://www.timeseriesclassification.com/dataset.php
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

    path = pathlib.Path(path) / "UCR_univariate"
    download_dataset(
        path,
        {
            "Univariate2018_arff.zip": "http://www.timeseriesclassification.com/Downloads/Archives/Univariate2018_arff.zip"
        },
        extract=True,
    )

    path = path / "extracted_Univariate2018_arff/Univariate_arff"

    datasets = list(path.glob("*"))
    all_data = {}

    todo = []
    for name in datasets:
        name = pathlib.Path(name)
        if name.is_file():
            continue
        todo.append(name)

    # create and configure the process pool
    with Pool(num_workers) as pool:
        # issues tasks to process pool
        results = tqdm(
            pool.imap(_loader, todo),
            desc="Loading UCR univariate...",
            total=len(datasets),
        )
        # iterate results
        for result in tuple(results):
            if result[0] is not None:
                all_data[result[0]] = result[1]
    return all_data


def _loader(name):
    data = name.name
    try:
        if (name / f"{data}_TRAIN.arff").is_file():
            data_train = arff.loadarff(name / f"{data}_TRAIN.arff")
            data_test = arff.loadarff(name / f"{data}_TEST.arff")

            data_train = np.asarray(
                [data_train[0][name] for name in data_train[1].names()]
            )
            X_train = data_train[:-1].T.astype("float64")
            y_train = data_train[-1].astype("int")

            data_test = np.asarray(
                [data_test[0][name] for name in data_test[1].names()]
            )
            X_test = data_test[:-1].T.astype("float64")
            y_test = data_test[-1].astype("int")
        else:
            data_train = np.genfromtxt(name / f"{data}_TRAIN.txt")
            data_test = np.genfromtxt(name / f"{data}_TEST.txt")

            X_train, y_train = data_train[:, 1:], data_train[:, 0].astype("int")
            X_test, y_test = data_test[:, 1:], data_test[:, 0].astype("int")
    except Exception as e:
        print(e, "for dataset:", data, "... SKIPPING")
        return (None, None)
    return (data, ((X_train, y_train), (X_test, y_test)))
