#!/usr/bin/env python

from .audiomnist import AudioMNIST
from .birdvox_dcase_20k import BirdVoxDCASE20k
from .CatsDogs import CatsDogs
from .freefield1010 import Freefield1010
from .fsd_kaggle_2018 import FSDKaggle2018
from .groove_MIDI import GrooveMIDI
from .gtzan import GTZAN
from .JapaneseVowels import JapaneseVowels
from .MosquitoSound import MosquitoSound
from .Phoneme import Phoneme
from .picidae import Picidae
from .seizures_neonatal import SeizuresNeonatal
from .sonycust import SONYCUST
from .speech_commands import SpeechCommands
from .UrbanSound import UrbanSound
from .vocalset import VocalSet
from .VoiceGenderDetection import VoiceGenderDetection
from .warblr import Warblr


__all__ = [
    "AudioMNIST",
    "BirdVoxDCASE20k",
    "CatsDogs",
    "Freefield1010",
    "FSDKaggle2018",
    "GrooveMIDI",
    "GTZAN",
    "JapaneseVowels",
    "MosquitoSound",
    "Phoneme",
    "Picidae",
    "SeizuresNeonatal",
    "SONYCUST",
    "SpeechCommands",
    "UrbanSound",
    "VoiceGenderDetection",
    "VocalSet",
    "Warblr",
]

# from . import (
#    VoiceGenderDetection,
#    JapaneseVowels,
#    UrbanSound,
#    Phoneme,
#    MosquitoSound,
#    CatsDogs,
#    UCR_univariate,
#    UCR_multivariate,
#    audiomnist,
#    speech_commands,
#    picidae,
#    birdvox_70k,
#    birdvox_dcase_20k,
#    esc,
#    gtzan,
#    irmas,
#    freefield1010,
#    warblr,
#    sonycust,
#    TUTacousticscenes2017,
#    groove_MIDI,
#    dcase_2019_task4,
#    vocalset,
#    seizures_neonatal,
# )

# https://github.com/ivclab/Sound20/tree/master/spectrogram_data
# https://www.gbif.org/dataset/b7ec1bf8-819b-11e2-bad2-00145eb45e9a
# https://github.com/YashNita/Animal-Sound-Dataset
# pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat', sep="\t", names = ['Frequency','Angle of attack','Chord length','Free-stream velocity','Suction/side','Scaled/sound'])
# https://dagshub.com/DagsHub/audio-datasets/src/main/voice_gender_detection
