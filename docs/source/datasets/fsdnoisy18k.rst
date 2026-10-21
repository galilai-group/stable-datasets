FSDnoisy18k
===========

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Sound%20Event%20Classification-blue" alt="Task: Sound Event Classification">
   <img src="https://img.shields.io/badge/Classes-20-green" alt="Classes: 20">
   <img src="https://img.shields.io/badge/Audio-44.1%20kHz%20mono-orange" alt="Audio: 44.1 kHz mono">
   </p>

Overview
--------

FSDnoisy18k is a sound event classification dataset for studying **real-world label noise**. It contains
**18,532 Freesound clips (42.5 h)** of 0.3-30 s, each with a **single label** from 20 classes of the
`AudioSet Ontology <https://research.google.com/audioset/ontology/index.html>`_.

- **train**: 17,585 clips
   - *clean* subset: 1,772 clips whose labels were manually verified (``manually_verified=True``)
   - *noisy* subset: 15,813 clips labelled only from Freesound tags. A manual audit of 15% of them found
     about 60% of the labels to be incorrect or incomplete (20-80% depending on the class).
     ``noisy_small=True`` marks 1,003 noisy clips, matching the clean subset in duration
- **test**: 947 clips, drawn entirely from the clean portion

Because the subsets are exposed as boolean columns rather than separate splits, the usual setups are a
filter away: clean-only training, noisy-only training, or the full train set.

Data Structure
--------------

.. list-table::
   :header-rows: 1
   :widths: 22 25 53

   * - Key
     - Type
     - Description
   * - ``series``
     - list[list[float]] (float32)
     - Mono waveform ``[T, 1]`` (``[T, C]`` like the other audio builders) on the raw 16-bit PCM scale
       (divide by 32768 for ``[-1, 1]``)
   * - ``sample_rate``
     - int
     - Sample rate of ``series`` (44100, or the config's rate)
   * - ``label``
     - int
     - Class id in ``[0, 19]``; names via ``ds.features["label"].names`` (sorted alphabetically)
   * - ``filename``
     - str
     - Source file name, e.g. ``"94322.wav"``
   * - ``freesound_id``
     - int
     - Freesound sound id
   * - ``manually_verified``
     - bool
     - ``True`` for the clean portion (all test clips, 1,772 train clips)
   * - ``noisy_small``
     - bool
     - ``True`` for the ``noisy_small`` train subset
   * - ``license``
     - str
     - Per-clip Creative Commons license URL

Configs
-------

``"native"`` (44.1 kHz, default, ~30 GB processed cache), ``"22khz"`` (~21 GB) and ``"16khz"`` (~15 GB);
resampling happens once at build time with a polyphase filter, and each config has its own cache. The raw
download is 9.5 GB and is verified against Zenodo's md5 checksums.

Usage Example
-------------

.. code-block:: python

    import numpy as np
    from stable_datasets.timeseries import FSDnoisy18k

    train = FSDnoisy18k(split="train", config_name="22khz")
    test = FSDnoisy18k(split="test", config_name="22khz")

    sample = train[0]
    wave = np.asarray(sample["series"], dtype=np.float32)[:, 0] / 32768.0   # [T] in [-1, 1]
    print(sample["sample_rate"], train.features["label"].names[sample["label"]], sample["manually_verified"])

Validation: SimCLR (Fonseca et al., 2021)
-----------------------------------------

``examples/fsd_simclr.py`` reproduces the linear evaluation of `Unsupervised Contrastive Learning of Sound
Event Representations <https://arxiv.org/abs/2011.07616>`_ on top of this builder, following the authors'
`code <https://github.com/edufonseca/uclser20>`_: log-mel patches of 1 s, SimCLR with mix-back on the noisy
train subset without labels, then a linear classifier on the frozen ResNet-18 trained on the noisy subset,
selected on the clean subset and evaluated once on the test set (balanced clip accuracy).

.. list-table:: Linear evaluation, balanced test accuracy (%), seeds 0-2
   :header-rows: 1
   :widths: 50 25 25

   * - Encoder
     - Mean ± std
     - Paper
   * - SimCLR pretrained with this example
     - 73.4 ± 0.8
     - 74.3
   * - Authors' released checkpoint, through this pipeline
     - 73.7 ± 0.4
     - 74.3
   * - Random initialisation
     - 35.3 ± 1.5
     - --

The released checkpoint isolates the data and evaluation side (builder, subsets, features, probe) from
pretraining. Per-seed metrics, the full protocol and versions are in ``fsdnoisy18k_simclr_results.json``.
One seed of pretraining takes about 2.5 h on an Apple M2 Max.

.. code-block:: bash

    python examples/fsd_simclr.py features --out runs/fsdnoisy18k
    python examples/fsd_simclr.py pretrain --features runs/fsdnoisy18k --out runs/fsdnoisy18k/simclr_seed0 --seed 0
    python examples/fsd_simclr.py probe --features runs/fsdnoisy18k \
        --encoder runs/fsdnoisy18k/simclr_seed0/encoder.pt --out runs/fsdnoisy18k/simclr_seed0/probe --seed 0

References
----------

- Zenodo record: https://zenodo.org/records/2529934
- Paper: https://arxiv.org/abs/1901.01189
- License: CC BY 4.0 for the dataset; each clip keeps its own Freesound CC license (see ``license``).

Citation
--------

.. code-block:: bibtex

    @inproceedings{fonseca2019learning,
      title     = {Learning Sound Event Classifiers from Web Audio with Noisy Labels},
      author    = {Fonseca, Eduardo and Plakal, Manoj and Ellis, Daniel P. W. and Font, Frederic and Favory, Xavier and Serra, Xavier},
      booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
      year      = {2019}
    }
