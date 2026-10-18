UCI HAR and MHEALTH
==================

These builders expose wearable-sensor time series using the existing download
and Arrow-cache interfaces. Both preserve source preprocessing and signal units.
They do not standardize the signals or apply training augmentations.

UCI HAR
-------

``UCIHAR`` returns the official subject-disjoint splits:

* ``train``: 7,352 windows from 21 subjects.
* ``test``: 2,947 windows from 9 different subjects.

Each example has these fields:

* ``series``: 128 time steps by 9 channels, stored as float32.
* ``label``: integer 0-5 (the source label minus one).
* ``subject_id``: the original subject number, 1-30.

The channel order is body acceleration x/y/z, body angular velocity x/y/z,
then total acceleration x/y/z. Acceleration is in g and angular velocity is in
rad/s. All signals are sampled at 50 Hz. The source already applies filtering
and uses 2.56-second windows with 50% overlap. The 561 engineered features are
not loaded.

Labels 0-5 correspond to walking, walking upstairs, walking downstairs, sitting,
standing, and laying. The exact channel order is also available as
``UCIHAR.CHANNEL_NAMES``.

MHEALTH
-------

``MHEALTH`` returns one unsplit collection, ``all``, with ten examples: one
continuous recording per subject. This collection is not a training split.
There is no official train/validation/test partition in the supplied archive.

Each example has these fields:

* ``series``: T time steps by 23 channels, stored as float32.
* ``label``: T aligned integer labels, preserving source values 0-12.
* ``subject_id``: the original subject number, 1-10.

Both ECG channels, all null-label samples, and the original sample order are
retained. There is no windowing, resampling, or concatenation across subjects.
The sampling rate is 50 Hz. The 23 channels are:

* Chest acceleration x/y/z; ECG leads 1 and 2.
* Left-ankle acceleration x/y/z, gyroscope x/y/z, magnetometer x/y/z.
* Right-lower-arm acceleration x/y/z, gyroscope x/y/z, magnetometer x/y/z.

The exact order is available as ``MHEALTH.CHANNEL_NAMES``. Source units are
m/s squared for acceleration, degrees/s for gyroscopes, local units for magnetic
field, and mV for ECG. The source README repeats column number 13; the ordered
magnetometer triplet occupies columns 12-14 (one-based).

Label 0 means the null class. Labels 1-12 are standing still, sitting and
relaxing, lying down, walking, climbing stairs, waist bends forward, frontal
elevation of arms, knees bending, cycling, jogging, running, and jumping front
and back.

Loading examples
----------------

Use ``STABLE_DATASETS_CACHE_DIR`` to choose a location for both downloads and
processed data. Alternatively, pass ``download_dir`` and
``processed_cache_dir`` to each builder.

.. code-block:: python

   import numpy as np
   from stable_datasets.timeseries import MHEALTH, UCIHAR

   har = UCIHAR(split="train")
   sample = har[0]
   x = np.asarray(sample["series"], dtype=np.float32)
   assert x.shape == (128, 9)
   print(sample["label"], sample["subject_id"])

   recordings = MHEALTH(split="all")
   record = recordings[0]
   signals = np.asarray(record["series"], dtype=np.float32)
   labels = np.asarray(record["label"], dtype=np.int64)
   assert signals.shape == (len(labels), 23)
   print(record["subject_id"], signals.shape)

The nested sequence fields use the library's list representation after cache
loading; the examples explicitly convert them to NumPy. For PyTorch, use
``torch.as_tensor(sample["series"], dtype=torch.float32)``. Convolutional models
usually expect channel-first tensors, so transpose the time and channel axes
in the training code.

Run ``python examples/activity_datasets.py`` for a loading example.

Training and evaluation example
-------------------------------

``examples/activity_simclr.py`` runs the existing ``stable-pretraining`` SimCLR
forward and NT-Xent loss with a small one-dimensional convolutional encoder.
This is a time-series adaptation, not a reproduction of the original image
benchmark. It compares frozen representations from an untrained encoder with
those from the same encoder after contrastive training.

The example uses the following fixed, subject-disjoint protocol:

* UCI HAR: keep the official test set. Reserve training subjects 27, 28, 29,
  and 30 for validation; use the other 17 official training subjects for
  training. The resulting counts are 5,867 / 1,485 / 2,947 windows.
* MHEALTH: subjects 1-6 train, 7-8 validation, 9-10 test. This is a custom
  partition, not an official benchmark split. Use 128-sample windows and
  stride 64, starting at sample zero on each original recording's timeline.
  Retain a window only if all 128 labels are equal and nonzero. Never join
  separated activity segments or cross subject boundaries. The counts are
  3,158 / 1,022 / 1,033 windows.
* MHEALTH classification uses the 21 motion channels, excluding ECG columns
  4 and 5 (one-based). The example maps activity labels 1-12 to 0-11;
  the dataset builder continues to preserve all 23 channels and labels 0-12.

**The MHEALTH selection uses ground-truth labels to remove null and transition
windows, including during pretraining.** Labels are not passed to the
contrastive loss, but this is not a fully annotation-free pipeline. The same
selection rule applies to all three splits and both representation comparisons.
The results concern curated activity windows, not continuous recognition with a
null class. Overlapping windows are correlated; splitting by subject prevents
same-subject overlap from leaking between training and evaluation.

Fit channel mean and standard deviation on training windows only, counting
overlap as represented in those windows. Apply these fixed statistics to
validation and test windows. Each training view independently applies a
per-channel scale sampled uniformly from 0.9 to 1.1 and Gaussian noise with
standard deviation 0.02 in normalized units. No time reversal, channel
permutation, cropping, or validation/test augmentation is used.

The encoder has three Conv1d-BatchNorm-ReLU blocks with 32, 64, and 128 output
channels, kernel sizes 7, 5, and 3, stride 2, and symmetric padding. Global
average pooling produces a 128-dimensional representation. A 128-128-64 MLP
with intermediate BatchNorm and ReLU is used only for contrastive training.
Training uses NT-Xent temperature 0.5, AdamW with learning rate 0.001 and weight
decay 0.0001, cosine decay per optimizer step, batch size 128, and 50 fixed
epochs. Incomplete training batches are dropped after each shuffled epoch.
The final epoch is evaluated; no test-guided checkpoint selection is performed.

For each encoder, fit a feature StandardScaler and L2-regularized multinomial
logistic regression on training embeddings only. Select C from
``[0.01, 0.1, 1, 10]`` using validation macro-F1 (first/smaller C wins a tie).
Evaluate that selected classifier on the test set without refitting on
validation subjects. Report accuracy and macro-F1, with per-class F1 and
confusion matrices saved in the run output. Seeds 0, 1, and 2 vary initialization,
shuffling, and augmentation; subject partitions stay fixed. The untrained
comparison starts from exactly the same encoder weights as its paired SimCLR
run, with default BatchNorm running statistics. It still uses training labels
to fit its linear classifier, so it is not a chance baseline.

Reproduction
~~~~~~~~~~~~

From the repository root, in a separate Python environment:

.. code-block:: bash

   python -m pip install -e '.[benchmarks]'
   python -m pip install 'stable-pretraining @ git+https://github.com/galilai-group/stable-pretraining.git@ab836bf699a2be712dfcf980a5eb70d36391e876'
   export STABLE_DATASETS_CACHE_DIR=/path/to/activity-cache
   python examples/activity_simclr.py --dataset uci_har --output /path/to/results/uci_har
   python examples/activity_simclr.py --dataset mhealth --output /path/to/results/mhealth

The recorded runs use CPU with four PyTorch threads. Output directories must
not already exist, to avoid overwriting a run. For a pipeline check, use
``--epochs 1 --seeds 0`` with a separate output directory; its metrics are not
part of the 50-epoch results.

Each output contains configuration and dependency versions, the executed script
and its SHA-256, source archive checksum, split manifests, training-only signal
statistics, epoch losses, initial and trained encoder weights, numeric linear
probe parameters, test predictions, and per-seed metrics. In a manifest,
``source_index`` is the zero-based window index within the official UCI HAR
source split, or the starting sample within the MHEALTH subject recording.
The raw datasets, caches, and binary experiment outputs are not committed.

Offline regression tests (no dataset download) can be run with:

.. code-block:: bash

   python -m pytest stable_datasets/tests/timeseries -q

Recorded results
~~~~~~~~~~~~~~~~

These runs completed 50 epochs for each seed (0, 1, 2) on macOS ARM64 CPU.
Values below are test percentages, mean +/- sample standard deviation over
the three seeds. Each representation uses its own validation-selected linear
classifier. Subject partitions are identical across seeds.

.. list-table:: Frozen linear evaluation
   :header-rows: 1

   * - Dataset
     - Encoder
     - Accuracy (%)
     - Macro-F1 (%)
   * - UCI HAR
     - Untrained + linear classifier
     - 88.47 +/- 1.90
     - 88.45 +/- 1.92
   * - UCI HAR
     - SimCLR + linear classifier
     - 90.99 +/- 0.54
     - 91.03 +/- 0.50
   * - MHEALTH
     - Untrained + linear classifier
     - 80.28 +/- 6.82
     - 78.04 +/- 9.11
   * - MHEALTH
     - SimCLR + linear classifier
     - 95.77 +/- 3.14
     - 95.32 +/- 3.28

Both datasets show higher mean accuracy and macro-F1 after SimCLR in these
runs. This is a small feasibility experiment, not a state-of-the-art claim
or a test of generalization across alternative subject partitions. The
MHEALTH test set contains only two subjects, and its label-informed window
selection excludes null and transition periods. Three training seeds do not
measure uncertainty over subjects or independent datasets.

The saved encoder and classifier parameters were reloaded independently;
they reproduced every recorded test prediction for both comparisons and
all six runs. Training losses and weights were finite, each run completed
50 epochs, and BatchNorm update counts matched the expected training steps.
All 23 offline time-series tests passed. The full repository test suite and
a rendered Sphinx documentation build were not run for this change.

Download the :download:`configuration, per-seed results, confusion matrices,
and verification summary <activity_simclr_results.json>`.

Recorded versions: Python 3.13.9, PyTorch 2.14.1, NumPy 2.5.4,
scikit-learn 1.9.1, Lightning 2.6.6. The exact
``stable-pretraining`` commit and executed script hash are in the JSON.

Sources and licenses
--------------------

* `UCI HAR official page <https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones>`_
  and `dataset DOI <https://doi.org/10.24432/C54S4K>`_.
* `MHEALTH official page <https://archive.ics.uci.edu/dataset/319/mhealth+dataset>`_
  and `dataset DOI <https://doi.org/10.24432/C5TW22>`_.

MHEALTH's current UCI page lists CC BY 4.0.

UCI HAR's current UCI page lists CC BY 4.0, while its bundled README prohibits
commercial use. This discrepancy is recorded in the builder metadata and
requires clarification before finalizing the license description; neither
statement is silently discarded.

Both builders download official UCI ZIP files and specify SHA-256 checksums.
Raw datasets are not included in this repository.

References
----------

* Anguita, D., Ghio, A., Oneto, L., Parra, X., and Reyes-Ortiz, J. L. (2013).
  A Public Domain Dataset for Human Activity Recognition Using Smartphones.
  ESANN 2013.
* Banos, O., Garcia, R., Holgado-Terriza, J. A., Damas, M., Pomares, H., Rojas, I.,
  Saez, A., and Villalonga, C. (2014). mHealthDroid: A Novel Framework for Agile
  Development of Mobile Health Applications. IWAAL 2014.

* Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. (2020).
  `A Simple Framework for Contrastive Learning of Visual Representations
  <https://proceedings.mlr.press/v119/chen20j.html>`_. ICML 2020.
