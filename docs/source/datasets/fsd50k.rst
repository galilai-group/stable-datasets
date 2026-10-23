FSD50K
======

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Multi--label%20Sound%20Event%20Recognition-blue" alt="Task: Multi-label Sound Event Recognition">
   <img src="https://img.shields.io/badge/Classes-200-green" alt="Classes: 200">
   <img src="https://img.shields.io/badge/Audio-44.1%20kHz%20mono-orange" alt="Audio: 44.1 kHz mono">
   </p>

Overview
--------

FSD50K is an open dataset of **51,197 Freesound clips (108 h)** of human-labeled sound events, annotated
with **200 classes** of the `AudioSet Ontology <https://research.google.com/audioset/ontology/index.html>`_
(144 leaf nodes and 56 intermediate nodes). Clips are 0.3-30 s long and **multi-labelled** (3.0 labels per
clip on average, after smearing). It was built as an open alternative to AudioSet: the audio itself is Creative Commons
licensed and hosted on Zenodo, so the dataset is fully reproducible.

- **train**: 36,796 clips (dev set, official ``split == "train"``)
- **validation**: 4,170 clips (dev set, official ``split == "val"``)
- **test**: 10,231 clips (the eval set; labels are exhaustive, i.e. correct and complete)

Dev and eval clips come from disjoint Freesound uploaders. Labels are **smeared** as released: every clip
also carries all ancestors of its labels up to the ontology root (e.g. ``Bark`` implies ``Dog``,
``Domestic_animals_and_pets`` and ``Animal``).

Data Structure
--------------

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

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
   * - ``labels``
     - list[int]
     - Class ids in ``[0, 199]`` (multi-label); names via ``ds.features["labels"].feature.names``
   * - ``mids``
     - list[str]
     - AudioSet ontology MIDs, aligned with ``labels``
   * - ``filename``
     - str
     - Source file name, e.g. ``"64760.wav"``
   * - ``freesound_id``
     - int
     - Freesound sound id
   * - ``license``
     - str
     - Per-clip Creative Commons license URL (CC0, CC-BY, CC-BY-NC or Sampling+)
   * - ``uploader``
     - str
     - Freesound user who uploaded the clip (for attribution)

Configs
-------

The waveform can be stored at the source rate or resampled once at build time (polyphase filter). Each
config has its own processed cache.

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Config
     - Sample rate
     - Approx. processed cache
   * - ``"native"`` (default)
     - 44,100 Hz
     - ~75 GB
   * - ``"22khz"``
     - 22,050 Hz
     - ~53 GB
   * - ``"16khz"``
     - 16,000 Hz
     - ~39 GB

The raw download is 24.6 GB. The audio is distributed as **split ZIP archives**
(``FSD50K.dev_audio.z01`` ... ``.zip``); the builder reads clips directly from the parts, so no merging or
extraction is needed. All files are verified against Zenodo's md5 checksums.

Usage Example
-------------

.. code-block:: python

    import numpy as np
    from stable_datasets.timeseries import FSD50K

    ds = FSD50K(split="train", config_name="16khz")
    sample = ds[0]

    wave = np.asarray(sample["series"], dtype=np.float32)[:, 0] / 32768.0   # [T] in [-1, 1]
    names = ds.features["labels"].feature.names
    print(sample["sample_rate"], [names[i] for i in sample["labels"]])

    # Multi-hot target for a 200-way sigmoid classifier
    target = np.zeros(len(names), dtype=np.float32)
    target[sample["labels"]] = 1.0

    # All splits at once
    splits = FSD50K(split=None)   # {"train", "validation", "test"}

Evaluation follows the paper: mean average precision (mAP), d' and lwlrap on ``test``, with model selection
on ``validation``.

Validation: supervised VGG-like baseline (Fonseca et al., 2022)
---------------------------------------------------------------

``examples/fsd50k_supervised.py`` reproduces the VGG-like row of the paper's baseline table (Table VI) on
top of the ``22khz`` config: 96-band log-mel patches of 1 s, a 0.27M-weight VGG-like network with 200
sigmoid outputs trained from scratch with binary cross-entropy, model selection on ``validation`` and a
single evaluation on ``test``. Clip scores are the mean of the patch outputs.

.. list-table:: Eval set (``test``), seeds 0-2, mean ± std
   :header-rows: 1
   :widths: 40 30 30

   * - Metric
     - This example
     - Paper
   * - mAP
     - 0.424 ± 0.005
     - 0.434 ± 0.002
   * - d'
     - 2.243 ± 0.015
     - 2.167 ± 0.011
   * - lwlrap (averaged over classes)
     - 0.521 ± 0.008
     - 0.514 ± 0.003

The paper's baseline code is no longer online, so a few details follow the text and Keras defaults; the
largest is that the checkpoint is selected on validation mAP instead of TensorFlow's PR-AUC. The paper
averages every metric over classes, including lwlrap. The DCASE definition of lwlrap, which weighs every
(clip, label) pair equally, gives 0.618 ± 0.006 on the same predictions, because frequent classes are
easier; the script reports both (``lwlrap_class`` and ``lwlrap``). Per-seed metrics, the full protocol and
versions are in ``fsd50k_supervised_results.json``. One seed takes 1.5-2.7 h on an RTX 2080 Ti
(about 4.6 min per epoch, early stopping after 21-35 epochs).

.. code-block:: bash

    python examples/fsd50k_supervised.py features --out runs/fsd50k
    python examples/fsd50k_supervised.py train --features runs/fsd50k --out runs/fsd50k/vgg_seed0 --seed 0

References
----------

- Zenodo record: https://zenodo.org/records/4060432
- Paper: https://arxiv.org/abs/2010.00475
- License: CC BY 4.0 for the dataset; each clip keeps its own Freesound CC license (see ``license``). CC-BY
  and Sampling+ clips require attribution to the ``uploader``; CC-BY-NC clips (~12%) forbid commercial use.

Citation
--------

.. code-block:: bibtex

    @article{fonseca2022fsd50k,
      title   = {{FSD50K}: An Open Dataset of Human-Labeled Sound Events},
      author  = {Fonseca, Eduardo and Favory, Xavier and Pons, Jordi and Font, Frederic and Serra, Xavier},
      journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
      volume  = {30},
      pages   = {829--852},
      year    = {2022}
    }
