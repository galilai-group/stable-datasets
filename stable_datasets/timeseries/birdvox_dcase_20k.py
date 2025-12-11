#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Randall Balestriero"

import io
import os
import pickle, gzip
import urllib.request
import numpy as np
import time
import zipfile
from tqdm import tqdm
from scipy.io.wavfile import read as wav_read
from ..utils import download_dataset


_urls = {
    "https://zenodo.org/record/1208080/files/BirdVox-DCASE-20k.zip?download=1": "BirdVox-DCASE-20k.zip",
    "https://ndownloader.figshare.com/files/10853300": "data_labels.csv",
}

_name = "birdvox_dcase_20k"


cite = """
@inproceedings{lostanlen2018icassp,
    title = {BirdVox-full-night: a dataset and benchmark for avian
    flight call detection},
    author = {Lostanlen, Vincent and Salamon, Justin and Farnsworth,
    Andrew and Kelling, Steve and Bello, Juan Pablo},
    booktitle = {Proc. IEEE ICASSP},
    year = {2018},
    published = {IEEE},
    venue = {Calgary, Canada},
    month = {April},
    }
    """


def load(path=None):
    """Binary bird detection classification

    Dataset is 16.5Go compressed.

    BirdVox-DCASE-20k: a dataset for bird audio detection in 10-second
    clips

    Version 2.0, March 2018.

    `link <https://wp.nyu.edu/birdvox>`_

    Description

    The BirdVox-DCASE-20k dataset contains 20,000 ten-second audio
    recordings. These recordings come from ROBIN autonomous recording
    units, placed near Ithaca, NY, USA during the fall 2015. They were
    captured on the night of September 23rd, 2015, by six different
    sensors, originally numbered 1, 2, 3, 5, 7, and 10.

    Out of these 20,000 recording, 10,017 (50.09%) contain at least one
    bird vocalization (either song, call, or chatter).

    The dataset is a derivative work of the BirdVox-full-night dataset
    [1], containing almost as much data but formatted into ten-second
    excerpts rather than ten-hour full night recordings.

    In addition, the BirdVox-DCASE-20k dataset is provided as a
    development set in the context of the "Bird Audio Detection"
    challenge, organized by DCASE (Detection and Classification of
    Acoustic Scenes and Events) and the IEEE Signal Processing Society.

    The dataset can be used, among other things, for the development and
    evaluation of bioacoustic classification models.


    We refer the reader to [1] for details on the distribution of the
    data and [2] for details on the hardware of ROBIN recording units.

    [1] V. Lostanlen, J. Salamon, A. Farnsworth, S. Kelling, J.P. Bello.
    "BirdVox-full-night: a dataset and benchmark for avian flight call
    detection", Proc. IEEE ICASSP, 2018.

    [2] J. Salamon, J. P. Bello, A. Farnsworth, M. Robbins, S. Keen,
    H. Klinck, and S. Kelling. Towards the Automatic Classification of
    Avian Flight Calls for Bioacoustic Monitoring. PLoS One, 2016.

    Data Files

    The wav folder contains the recordings as WAV files, sampled at
    44,1 kHz, with a single channel (mono). The original sample rate
    was 24 kHz.

    The name of each wav file is a random 128-bit UUID (Universal
    Unique IDentifier) string, which is randomized with respect to the
    origin of the recording in BirdVox-full-night, both in terms of
    time (UTC hour at the start of the excerpt) and space (location of
    the sensor).

    The origin of each 10-second excerpt is known by the challenge
    organizers, but not disclosed to the participants.

    Please Acknowledge BirdVox-DCASE-20k in Academic Research

    When BirdVox-70k is used for academic research, we would highly
    appreciate it if  scientific publications of works partly based on
    this dataset cite the following publication:

    V. Lostanlen, J. Salamon, A. Farnsworth, S. Kelling, J. Bello.
    "BirdVox-full-night: a dataset and benchmark for avian flight call
    detection", Proc. IEEE ICASSP, 2018.

    The creation of this dataset was supported by NSF grants 1125098
    (BIRDCAST) and 1633259 (BIRDVOX), a Google Faculty Award, the Leon
    Levy Foundation, and two anonymous donors.
    Parameters
    ----------
    path: str (optional)
        default ($DATASET_PATH), the path to look for the data and
        where the data will be downloaded if not present

    Returns
    -------

    wavs: array
        the waveforms in the time amplitude domain

    labels: array
        binary values representing the presence or not of an avian

    recording: array
        the file number from which the sample has been extracted

    """

    if path is None:
        path = os.environ["DATASET_PATH"]

    download_dataset(path, _name, _urls)

    t0 = time.time()

    # Loading the file
    basefile = os.path.join(path, "birdvox_dcase_20k/BirdVox-DCASE-20k.zip")
    wavs = list()
    labels = np.loadtxt(
        os.path.join(path, "birdvox_dcase_20k/data_labels.csv"),
        skiprows=1,
        delimiter=",",
        dtype="str",
    )
    wav_names = list(labels[:, 0])
    wav_labels = labels[:, 2].astype("int")
    labels = list()
    f = zipfile.ZipFile(basefile)
    for name in tqdm(f.namelist(), ascii=True):
        filename = name.split("/")[-1][:-4]
        if ".wav" not in name or filename not in wav_names:
            continue
        byt = io.BytesIO(f.read(name))
        wavs.append(wav_read(byt)[1].astype("float32"))
        labels.append(wav_labels[wav_names.index(filename)])

    wavs = np.array(wavs).astype("float32")
    labels = np.array(labels).astype("int32")

    print("Dataset birdvox_dcase_20k loaded in {0:.2f}s.".format(time.time() - t0))
    dataset = {"wavs": wavs, "labels": labels}
    return dataset
