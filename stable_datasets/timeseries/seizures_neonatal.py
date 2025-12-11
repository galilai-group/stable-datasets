import os
import urllib.request
import time
import io
from scipy.io import loadmat

# import mne
from tqdm import tqdm
from ..utils import download_dataset


_urls = {
    "https://zenodo.org/record/2547147/files/annotations_2017.mat?download=1": "annotations_2017.mat"
}
_urls.update(
    {
        "https://zenodo.org/record/2547147/files/eeg{}.edf?download=1".format(
            i
        ): "eeg{}.edf".format(i)
        for i in range(1, 80)
    }
)

_name = "seizures_neonatal"


def load(path=None):
    """A dataset of neonatal EEG recordings with seizures annotations

    source: https://zenodo.org/record/2547147

    Neonatal seizures are a common emergency inthe neonatal intensive care unit
    (NICU). There are many questions yet to be answered regarding the
    temporal/spatial characteristics of seizures from different pathologies,
    response to medication, effects on neurodevelopment and optimal detection.
    This dataset contains EEG recordings from human neonates and the visual
    interpretation of the EEG by the human expert. Multi-channel EEG was
    recorded from 79 term neonates admitted to the neonatal intensive care unit
    (NICU) at the Helsinki University Hospital. The median recording duration
    was 74 minutes (IQR: 64 to 96 minutes). EEGs were annotated by three experts
    for the presence of seizures. An average of 460 seizures were annotated per
    expert in the dataset, 39 neonates had seizures by consensus and 22 were
    seizure free by consensus. The dataset can be used as a reference set of
    neonatal seizures, for the development of automated methods of seizure
    detection and other EEG analysis, as well as for the analysis of
    inter-observer agreement.
    Parameters
    ----------

    path: str (optional)
        a string where to load the data and download if not present

    Returns
    -------

    annotations: list
        the list of multichannel binary vectors representing
        the presence or absence of seizure, 3 channels due to
        3 expert annotations

    waveforms: list
        list of (channels, TIME) multichannel EEGs
    """
    if path is None:
        path = os.environ["DATASET_PATH"]

    download_dataset(path, _name, _urls)
    t = time.time()

    # load wavs
    annotations = loadmat(os.path.join(path, "seizures_neonatal/annotations_2017.mat"))
    annotations = annotations["annotat_new"][0]

    # init. the data array
    waveforms = []
    for i in tqdm(range(1, 80), ascii=True):
        filename = os.path.join(path, "seizures_neonatal/eeg{}.edf".format(i))
        data = mne.io.read_raw_edf(filename)
        waveforms.append(data.get_data())

    dataset = {"wavs": waveforms, "labels": annotations}
    return dataset
