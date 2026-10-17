QMNIST
======

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Image%20Classification-blue" alt="Task: Image Classification">
   <img src="https://img.shields.io/badge/Classes-10-green" alt="Classes: 10">
   <img src="https://img.shields.io/badge/Size-28x28-orange" alt="Image Size: 28x28">
   </p>

`QMNIST <https://github.com/facebookresearch/qmnist>`_ reconstructs the MNIST
digits from NIST Special Database 19. It adds the recovered 50,000 test digits
and provenance fields that associate samples with their source writers.
This builder downloads the authors' files at commit
``94061367ec2e099ea18c553c75727ecf849ddb9b``, with SHA-256 verification.

Splits and evaluation
---------------------

.. list-table::
   :header-rows: 1

   * - Split
     - Samples
     - Meaning
   * - ``train``
     - 60,000
     - Original QMNIST training order
   * - ``test``
     - 60,000
     - Complete QMNIST test set
   * - ``test10k``
     - 10,000
     - First 10,000 rows of ``test``
   * - ``test50k``
     - 50,000
     - Remaining 50,000 rows of ``test``

``test10k`` and ``test50k`` partition ``test``: do not concatenate all three
as independent observations. No validation split is invented. No samples are
filtered, shuffled, resized, normalized, or relabeled during ingestion.
Pixels remain 28 x 28 grayscale uint8, in the original sample order.

QMNIST reconstructs MNIST sample identities. In particular, do not treat MNIST
and QMNIST as independent data when mixing training and testing sources.
The authors' split is preserved; this builder does not claim writer-independent
evaluation. The larger ``xnist`` release is not included by this builder.

Usage
-----

.. code-block:: python

   from stable_datasets.images import QMNIST

   datasets = QMNIST()
   train = datasets["train"]
   extended_test = datasets["test50k"]
   row = train[0]
   print(row["image"].size, row["label"], row["nist_writer_id"])
   tensor = train.with_format("torch")[0]["image"]  # float32, (1, 28, 28), [0, 1]

As with the base builder, the first call prepares all splits. Subsequent calls
reuse their processed caches.

Fields
------

``image`` is a PIL grayscale image and ``label`` is the digit class (0--9).
The remaining seven integer fields retain columns 1--7 of the official
big-endian int32 extended label file:

* ``nist_hsf_series``: NIST HSF series.
* ``nist_writer_id``: NIST writer identifier.
* ``nist_digit_index``: digit index within the writer's data.
* ``nist_class_code``: original NIST class code.
* ``nist_global_index``: index into the authors' NIST digit reconstruction.
* ``duplicate``: official duplicate field (zero in the QMNIST train/test release).
* ``unused``: official unused field, retained without reinterpretation.

Validation
----------

.. code-block:: bash

   pytest stable_datasets/tests/images/test_q_mnist.py -m "not large"
   pytest stable_datasets/tests/images/test_q_mnist.py -m large
   python examples/validate_qmnist_omniglot.py qmnist --output qmnist_validation.json

The example trains a small supervised CNN on ``train`` and evaluates the final
model on ``test10k`` and ``test50k``. It records its configuration and results;
it is a loader-validation baseline, not a reproduction of the paper's LeNet-5
experiment or a claim of state-of-the-art accuracy.

License and citation
--------------------

The repository's
`BSD-3-Clause license <https://github.com/facebookresearch/qmnist/blob/94061367ec2e099ea18c553c75727ecf849ddb9b/LICENSE>`_
explicitly applies to "QMNIST software". A separate license for the image and
label files is not specified there. This builder downloads the authors' public
files; it does not redistribute them or assign them a new license.
Please cite Chhavi Yadav and Léon Bottou, *Cold Case: The Lost MNIST Digits*,
Advances in Neural Information Processing Systems 32 (2019).
