CIFAR-10.1 and CIFAR-10.2
=========================

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Distribution%20Shift%20Evaluation-blue" alt="Task: Distribution Shift Evaluation">
   <img src="https://img.shields.io/badge/Classes-10-green" alt="Classes: 10">
   <img src="https://img.shields.io/badge/Size-32x32x3-orange" alt="Image Size: 32x32x3">
   </p>

Overview
--------

``CIFAR10_1`` and ``CIFAR10_2`` are CIFAR-10-compatible image-classification
benchmarks for studying natural distribution shift. Both use the original
CIFAR-10 label space—airplane, automobile, bird, cat, deer, dog, frog, horse,
ship, and truck—so a model can be evaluated without changing its prediction
head.

CIFAR-10.1 is an independently collected test set designed to measure how well
models selected on the original CIFAR-10 benchmark generalize to fresh images.
CIFAR-10.2 reproduces the CIFAR-10 collection pipeline and includes train and
test splits for controlled distribution-shift experiments.

- **CIFAR-10.1**: 2,000 class-balanced test images.
- **CIFAR-10.2**: 10,000 training images and 2,000 test images.
- **Image resolution**: 32×32×3 RGB.
- **Classes**: 10, matching CIFAR-10 exactly.

Data Structure
--------------

When accessing an example using ``ds[i]``, both datasets return a dictionary
with the following keys:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Key
     - Type
     - Description
   * - ``image``
     - ``PIL.Image.Image``
     - A 32×32 RGB image.
   * - ``label``
     - ``int``
     - CIFAR-10 class label from 0 to 9.

CIFAR-10 Label Space
--------------------

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Label
     - Class Name
   * - 0
     - ``airplane``
   * - 1
     - ``automobile``
   * - 2
     - ``bird``
   * - 3
     - ``cat``
   * - 4
     - ``deer``
   * - 5
     - ``dog``
   * - 6
     - ``frog``
   * - 7
     - ``horse``
   * - 8
     - ``ship``
   * - 9
     - ``truck``

Usage Example
-------------

**Basic Usage**

.. code-block:: python

    from stable_datasets.images import CIFAR10_1, CIFAR10_2

    # CIFAR-10.1 is a test-only dataset.
    cifar10_1_test = CIFAR10_1(split="test")

    # CIFAR-10.2 provides both train and test splits.
    cifar10_2_train = CIFAR10_2(split="train")
    cifar10_2_test = CIFAR10_2(split="test")

    sample = cifar10_1_test[0]
    print(sample.keys())  # {"image", "label"}
    print(sample["label"])  # Integer from 0 to 9

    # Optional: make images PyTorch-friendly.
    cifar10_2_train_torch = cifar10_2_train.with_format("torch")

**Getting Class Names**

.. code-block:: python

    from stable_datasets.images import CIFAR10_1

    ds = CIFAR10_1(split="test")
    class_names = ds.info.features["label"].names

    sample = ds[0]
    print(class_names[sample["label"]])

Typical Workflow: Distribution-Shift Evaluation
------------------------------------------------

Train a model on the original CIFAR-10 data, then evaluate the unchanged model
on CIFAR-10.1 to test generalization to independently collected images.
CIFAR-10.2 can instead be used as a self-contained train/test benchmark.

.. code-block:: python

    from stable_datasets.images import CIFAR10, CIFAR10_1, CIFAR10_2

    # Train on the original benchmark, then evaluate on fresh CIFAR-10.1 data.
    cifar10_train = CIFAR10(split="train").with_format("torch")
    cifar10_1_test = CIFAR10_1(split="test").with_format("torch")

    # Or train and evaluate solely within the CIFAR-10.2 release.
    cifar10_2_train = CIFAR10_2(split="train").with_format("torch")
    cifar10_2_test = CIFAR10_2(split="test").with_format("torch")

    # ... create DataLoaders, train a classifier, and compare test accuracy ...

Related Datasets
----------------

- :doc:`cifar10`: Original CIFAR-10 data for training and baseline comparison.
- :doc:`cifar100`: A related 100-class image-classification benchmark.
- :doc:`cifar10_c`: CIFAR-10 images with synthetic common corruptions.

References
----------

- CIFAR-10.1 repository: https://github.com/modestyachts/CIFAR-10.1
- CIFAR-10.2 repository: https://github.com/modestyachts/cifar-10.2
- CIFAR-10.1 paper: `Do CIFAR-10 Classifiers Generalize to CIFAR-10? <https://arxiv.org/abs/1806.00451>`_
- CIFAR-10.2 paper: `Harder or Different? A Closer Look at Distribution Shift in Dataset Reproduction <https://arxiv.org/abs/2106.04539>`_
- Data terms: both releases contain images derived from the 80 Million Tiny Images dataset.

Citation
--------

.. code-block:: bibtex

    @article{recht2018cifar10,
      author={Recht, Benjamin and Roelofs, Rebecca and Schmidt, Ludwig and Shankar, Vaishaal},
      title={Do CIFAR-10 Classifiers Generalize to CIFAR-10?},
      journal={arXiv preprint arXiv:1806.00451},
      year={2018}
    }

    @inproceedings{lu2022harder,
      author={Lu, Shangyun and Nott, Bradley and Olson, Aaron and Todeschini, Alberto and
              Vahabi, Puya and Carmon, Yair and Schmidt, Ludwig},
      title={Harder or Different? A Closer Look at Distribution Shift in Dataset Reproduction},
      booktitle={International Conference on Learning Representations},
      year={2022}
    }
