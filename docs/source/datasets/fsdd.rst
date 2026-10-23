Free Spoken Digit Dataset (FSDD)
================================

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Spoken%20Digit%20Classification-blue" alt="Task: Spoken Digit Classification">
   <img src="https://img.shields.io/badge/Classes-10-green" alt="Classes: 10">
   <img src="https://img.shields.io/badge/License-CC%20BY--SA%204.0-yellow" alt="License: CC BY-SA 4.0">
   </p>

Overview
--------

The Free Spoken Digit Dataset contains 3,000 English recordings of the digits
0--9 from six speakers. Each speaker recorded every digit 50 times. Recordings
are trimmed, mono, 8 kHz WAV files. The builder is fixed to source revision
``26eb9aaf76e81b692f806f9140c2d2777410d7a1`` so that the growing upstream
repository does not change an experiment after its cache has been created.

This dataset is distinct from AudioMNIST.

Official Splits
---------------

FSDD defines its split by the recording index in
``{digit}_{speaker}_{index}.wav``.

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - Split
     - Recording indices
     - Samples
     - Meaning
   * - ``train``
     - 5--49
     - 2,700
     - Official training set
   * - ``test``
     - 0--4
     - 300
     - Official test set

There is no official validation split. For the accompanying supervised
experiment, indices 5--9 are selected from ``train`` as an experiment-defined
validation set, and indices 10--49 remain available for fitting. The official
``test`` split is not used for model selection. Because every speaker occurs in
each subset, this protocol is not speaker-independent.

Data Structure
--------------

Each example contains the following keys.

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Key
     - Type
     - Description
   * - ``series``
     - ``float32 [T, 1]``
     - Mono waveform in signed 16-bit PCM sample units
   * - ``label``
     - int
     - Spoken digit, from 0 through 9
   * - ``speaker_id``
     - string
     - Speaker name parsed from the filename
   * - ``utterance_id``
     - int32
     - Recording index, from 0 through 49
   * - ``filename``
     - string
     - Original recording filename
   * - ``sample_rate``
     - int32
     - Sample rate read from the WAV header (8,000 Hz)

``series`` preserves the numeric values of the source 16-bit PCM samples when
casting them to ``float32``; it is not normalized to ``[-1, 1]``. Divide by
32,768 during preprocessing if normalized amplitudes are required. Recording
lengths vary, so batches must pad to the longest item and retain the true
lengths for masking.

Usage Example
-------------

.. code-block:: python

    from stable_datasets.timeseries import FSDD

    train = FSDD(split="train")
    test = FSDD(split="test")

    sample = train[0]
    waveform = sample["series"]
    digit = sample["label"]
    speaker = sample["speaker_id"]
    sample_rate = sample["sample_rate"]

    print(len(train), len(test))  # 2700, 300
    print(len(waveform), len(waveform[0]), sample_rate)  # T, 1, 8000

Experiment Reproduction
-----------------------

The project's ``benchmarks`` extra follows the current ``stable-pretraining``
Git source without pinning a revision. To reproduce the tracked reference run,
create a Python 3.12 environment and install the exact ``stable-pretraining``
revision plus the core package versions verified in the retained reference
environment:

.. code-block:: console

    python3.12 -m venv .venv
    . .venv/bin/activate
    python -m pip install -e . \
        "stable-pretraining @ git+https://github.com/galilai-group/stable-pretraining.git@ab836bf699a2be712dfcf980a5eb70d36391e876" \
        "torch==2.14.1" "torchvision==0.29.1" "lightning==2.6.6" \
        "numpy==2.5.4" "pillow==13.0.0"

Then run the deterministic supervised baseline:

.. code-block:: console

    python examples/validate_digits.py \
        --dataset fsdd \
        --epochs 15 \
        --batch-size 64 \
        --learning-rate 0.001 \
        --weight-decay 0.0001 \
        --seed 42 \
        --device cpu \
        --output-dir benchmark-runs/digit-validation

The script loads both official splits through ``FSDD``. It uses
``utterance_id`` 10--49 for fitting (2,400 recordings), 5--9 for validation
(300 recordings), and the untouched official test split 0--4 for final
evaluation (300 recordings). Waveforms are divided by 32,768, padded without
cropping, and accompanied by valid lengths. The 1D CNN propagates those lengths
through every convolution and excludes padding from masked mean and max
pooling. Validation accuracy selects the checkpoint.

The 2026-10-23 reference run used one seed, a 22,986-parameter CNN, AdamW, and
15 epochs on CPU. The selected epoch was 15: train loss/accuracy were
1.3876/46.5833%, validation loss/accuracy were 1.4169/44.0000%, and final
official-test loss/accuracy were 1.3971/47.0000%. Runtime was 90.34 seconds.
This deliberately small model is an above-chance pipeline validation; it is
not a tuned or speaker-independent benchmark. Its 47.00% test accuracy is a
weak baseline and should not be read as a state-of-the-art result.

The reported train loss and accuracy are training-loop aggregates from epoch
15. They were not recomputed by evaluating the restored best checkpoint on the
training split.

The run used Python 3.12.6, PyTorch 2.14.1, Lightning 2.6.6, and
``stable-pretraining`` commit
``ab836bf699a2be712dfcf980a5eb70d36391e876``
(``0.1.9.dev10+gab836bf69``). Metrics and manifests are stored under
``examples/results/digit_validation/fsdd``; checkpoints and dataset caches are
intentionally excluded from version control.

Dataset Details
---------------

- **Source**: https://github.com/Jakobovski/free-spoken-digit-dataset
- **Pinned revision**: ``26eb9aaf76e81b692f806f9140c2d2777410d7a1``
- **License**: Creative Commons Attribution-ShareAlike 4.0 International
- **Audio format**: mono, 8 kHz, signed 16-bit PCM WAV
- **Total recordings**: 3,000
- **Speakers**: 6

Citation
--------

.. code-block:: bibtex

    @misc{jackson2016fsdd,
        author = {Jackson, Zohar},
        title = {Free Spoken Digit Dataset},
        year = {2016},
        url = {https://github.com/Jakobovski/free-spoken-digit-dataset}
    }
