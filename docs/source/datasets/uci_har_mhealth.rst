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

Evaluation status
-----------------

Training and evaluation experiments are not implemented yet. MHEALTH windowing,
null-label handling, channel selection, and subject partitions remain to be
chosen for the training example. The loader deliberately preserves the data
needed to make these choices explicitly.

Split subjects before constructing training/evaluation windows, and fit any
normalization using training data only. Randomly splitting overlapping windows
from the same recording can leak information between training and evaluation.
For UCI HAR, keep the official test subjects separate and derive any validation
subset from the training subjects.

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
