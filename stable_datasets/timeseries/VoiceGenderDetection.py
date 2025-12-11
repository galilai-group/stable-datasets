import os
from ..utils import download_dataset
import pathlib
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm
from multiprocessing import Pool


def load(path=None, num_workers=16):
    """

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

    path = pathlib.Path(path) / "VoiceGenderDetection"
    download_dataset(
        path,
        {
            "VoxCeleb_gender.zip": "https://drive.google.com/u/0/uc?id=1HRbWocxwClGy9Fj1MQeugpR4vOaL9ebO&export=download"
        },
        extract=True,
    )

    path = path / "extracted_VoxCeleb_gender/VoxCeleb_gender/"

    males = list((path / "males").glob("*.m4a"))
    females = list((path / "females").glob("*.m4a"))

    with Pool(num_workers) as pool:
        X_train = tuple(
            tqdm(
                pool.imap(_reader, males),
                desc="Loading male voices...",
                total=len(males),
            )
        )
        y_train = np.zeros(len(X_train))

    with Pool(num_workers) as pool:

        results = tuple(
            tqdm(
                pool.imap(_reader, females),
                desc="Loading female voices...",
                total=len(females),
            )
        )
        # iterate results
        X_train += tuple(results)
    y_train = np.concatenate([y_train, np.ones(len(results))])

    return (X_train, y_train), None


def _reader(x):
    return AudioSegment.from_file(x).get_array_of_samples()
