import os
import numpy as np
from scipy.io import arff
from tqdm import tqdm
from ..utils import download_dataset, tolist_recursive
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

    path = pathlib.Path(path) / "UCR_multivariate"
    download_dataset(
        path,
        {
            "Multivariate2018_arff.zip": "http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_arff.zip"
        },
        extract=True,
    )

    path = path / "extracted_Multivariate2018_arff/Multivariate_arff"

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
            desc="Loading UCR multivariate...",
            total=len(datasets),
        )
        # iterate results
        for result in tuple(results):
            if result[0] is not None:
                all_data[result[0]] = result[1]
    return all_data


def _loader(name):
    files = list(name.glob("*TRAIN*"))
    data = name.name
    print(name)
    try:
        X_train, y_train = [], []
        X_test, y_test = [], []
        for subfile in files:
            ext = subfile.suffix
            assert ext == ".arff"

            data_train = arff.loadarff(subfile)
            data_train = np.asarray(data_train[0].tolist())
            data_test = arff.loadarff(str(subfile).replace("TRAIN", "TEST"))
            data_test = np.asarray(data_test[0].tolist())
            X_train.append(data_train[:, :-1])
            X_test.append(data_test[:, :-1])
            y_train.append(data_train[:, -1])
            y_test.append(data_test[:, -1])

            # data_train = arff.loadarff(subfile)[0]
            # data_test = arff.loadarff(str(subfile).replace("TRAIN", "TEST"))[0]

            # for n in range(len(data_train)):
            #     y_train.append(data_train[n][1])
            #     x = np.stack(
            #         [data_train[n][0][name] for name in data_train[n][0].names()], 1
            #     )
            #     X_train.append(x)
            # for n in range(len(data_test)):
            #     y_test.append(data_test[n][1])
            #     x = np.stack(
            #         [data_test[n][0][name] for name in data_test[n][0].names()], 1
            #     )
            #     X_test.append(x)

    except Exception as e:
        # raise (e)
        print(e, "for dataset:", data, "... SKIPPING")
        return (None, None)
    if len(X_train) == 0:
        print("Nothing for dataset:", data, "... SKIPPING")
        return (None, None)
    # X_train = np.concatenate(X_train)
    # X_test = np.concatenate(X_test)
    y_train = np.concatenate(y_train)
    y_test = np.concatenate(y_test)
    return (data, ((X_train, y_train), (X_test, y_test)))
