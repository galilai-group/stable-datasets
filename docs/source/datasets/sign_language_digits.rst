Sign Language Digits
====================

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Image%20Classification-blue" alt="Task: Image Classification">
   <img src="https://img.shields.io/badge/Classes-10-green" alt="Classes: 10">
   <img src="https://img.shields.io/badge/Size-Mostly%20100x100-orange" alt="Image Size: Mostly 100x100">
   </p>

Overview
--------

Sign Language Digits is an image-classification dataset of American Sign
Language hand signs for the digits 0--9. The fixed source revision contains
2,062 RGB images collected from students. Of these, 2,059 images are 100 x 100
pixels and three source images in class 7 are 3024 x 3024 pixels. The builder
preserves those source dimensions. This dataset is distinct from Sign Language
MNIST.

The builder uses source revision
``4b09433ef9d5c60602e8805bb54b8aa2428527f8`` and reads only files located
directly in ``Dataset/<digit>/``. Images in ``Examples/`` are previews and are
not dataset samples. The downloaded archive is verified against SHA-256 digest
``b002e3b3819b5ec270d3d424ef8c7d9b66372cb6b8f92fbfe3a258abb3b3ccb0``.
The data is distributed under the Apache-2.0 license.

Split
-----

The source repository does not publish an official train, validation, or test
partition. The builder therefore exposes the complete source image pool as the
single ``all`` split. Any partition used for an experiment is user-defined and
should be documented separately.

The class counts in the pinned revision, ordered from digit 0 through digit 9,
are ``[205, 206, 206, 206, 207, 207, 207, 206, 208, 204]``.

Data Structure
--------------

When accessing an example using ``ds[i]``, you receive a dictionary with the
following keys:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Key
     - Type
     - Description
   * - ``image``
     - ``PIL.Image.Image``
     - Source-resolution image decoded in RGB mode; most are 100 x 100
   * - ``label``
     - int
     - Digit class in the inclusive range 0--9
   * - ``filename``
     - str
     - Stable path relative to ``Dataset/``, for example ``0/IMG_1118.JPG``

Usage Example
-------------

.. code-block:: python

    from stable_datasets.images import SignLanguageDigits

    dataset = SignLanguageDigits(split="all")
    sample = dataset[0]

    print(len(dataset))          # 2062
    print(sample["filename"])
    print(sample["image"].mode)  # RGB
    print(sample["label"])       # 0 through 9

For a reproducible experiment, create stratified partitions from ``filename``
and save the resulting manifests. Such partitions are not official dataset
splits, and should not be described as signer-independent because the source
does not provide reliable signer identifiers for this purpose.

Experiment Reproduction
-----------------------

Install the ``benchmarks`` dependency group, then run the deterministic
supervised baseline:

.. code-block:: console

    python examples/validate_digits.py \
        --dataset sign-language-digits \
        --epochs 15 \
        --batch-size 64 \
        --learning-rate 0.001 \
        --weight-decay 0.0001 \
        --seed 42 \
        --device cpu \
        --output-dir benchmark-runs/digit-validation

The script loads the data through ``SignLanguageDigits``, resizes images to
64 x 64, maps pixel values to ``[-1, 1]``, and writes the exact stratified
split manifests. The resulting train/validation/test counts are
1,648/210/204. Validation accuracy selects the checkpoint; the test partition
is evaluated only after selection.

The 2026-10-23 reference run used one seed, a 156,298-parameter CNN, AdamW,
and 15 epochs on CPU. The selected epoch was 10: train loss/accuracy were
0.1037/97.9976%, validation loss/accuracy were 0.1971/94.7619%, and final test
loss/accuracy were 0.1426/94.6078%. Runtime was 88.37 seconds. This is a
lightweight pipeline check on an experiment-defined split, not a
signer-independent benchmark or a multi-seed performance claim.

The run used Python 3.12.6, PyTorch 2.14.1, Lightning 2.6.6, and
``stable-pretraining`` commit
``ab836bf699a2be712dfcf980a5eb70d36391e876``
(``0.1.9.dev10+gab836bf69``). Metrics and manifests are stored under
``examples/results/digit_validation/sign_language_digits``; checkpoints and
dataset caches are intentionally excluded from version control.

References
----------

- Source repository: https://github.com/ardamavi/Sign-Language-Digits-Dataset
- Pinned source revision: https://github.com/ardamavi/Sign-Language-Digits-Dataset/tree/4b09433ef9d5c60602e8805bb54b8aa2428527f8
- Paper: https://arxiv.org/abs/2011.08927

Citation
--------

.. code-block:: bibtex

    @article{mavi2020sign,
      title={A New Dataset and Proposed Convolutional Neural Network Architecture for Classification of American Sign Language Digits},
      author={Mavi, Arda},
      journal={arXiv preprint arXiv:2011.08927},
      year={2020}
    }
