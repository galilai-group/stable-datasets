CIFAR-10-C
==========

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Robustness%20Evaluation-blue" alt="Task: Robustness Evaluation">
   <img src="https://img.shields.io/badge/Classes-10-green" alt="Classes: 10">
   <img src="https://img.shields.io/badge/Size-32x32-orange" alt="Image Size: 32x32">
   <img src="https://img.shields.io/badge/Corruptions-19-red" alt="Corruptions: 19">
   </p>

Overview
--------

CIFAR-10-C is a corrupted version of the CIFAR-10 dataset designed for benchmarking neural network robustness to common corruptions and perturbations. It consists of the CIFAR-10 test set with 19 different types of corruptions applied at 5 severity levels each.

- **Test**: 950,000 images (19 corruptions × 5 severity levels × 10,000 images)
- **Train**: N/A (test-only dataset for robustness evaluation)

The dataset includes corruptions such as noise, blur, weather effects, and digital artifacts to evaluate how well models trained on clean CIFAR-10 data generalize to corrupted inputs.

Data Structure
--------------

When accessing an example using ``ds[i]``, you will receive a dictionary with the following keys:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Key
     - Type
     - Description
   * - ``image``
     - ``PIL.Image.Image``
     - 32×32×3 RGB corrupted image
   * - ``label``
     - int
     - Class label (0-9)
   * - ``corruption_name``
     - str
     - Type of corruption applied (e.g., "gaussian_noise", "fog")
   * - ``corruption_level``
     - int
     - Severity level of corruption (1-5, where 5 is most severe)

Corruption Types
----------------

The dataset includes 19 corruption types across different categories:

**Noise:**
- ``gaussian_noise``
- ``shot_noise``
- ``impulse_noise``
- ``speckle_noise``

**Blur:**
- ``defocus_blur``
- ``glass_blur``
- ``motion_blur``
- ``zoom_blur``
- ``gaussian_blur``

**Weather:**
- ``snow``
- ``frost``
- ``fog``

**Digital:**
- ``brightness``
- ``contrast``
- ``elastic_transform``
- ``pixelate``
- ``jpeg_compression``
- ``saturate``
- ``spatter``

Usage Example
-------------

**Basic Usage**

.. code-block:: python

    from stable_datasets.images.cifar10_c import CIFAR10C

    # Load the test set (only split available)
    ds = CIFAR10C(split="test")

    sample = ds[0]
    print(sample.keys())  # {"image", "label", "corruption_name", "corruption_level"}
    print(f"Corruption: {sample['corruption_name']} at level {sample['corruption_level']}")

    # Optional: make it PyTorch-friendly
    ds_torch = ds.with_format("torch")

**Typical Workflow**

.. code-block:: python

    from stable_datasets.images.cifar10 import CIFAR10
    from stable_datasets.images.cifar10_c import CIFAR10C

    # Train on clean CIFAR-10
    train_data = CIFAR10(split="train")
    # ... train your model ...

    # Evaluate robustness on CIFAR-10-C
    test_data = CIFAR10C(split="test")
    # ... evaluate on corrupted images ...

Related Datasets
----------------

- :doc:`cifar10`: Original clean CIFAR-10 dataset for training
- :doc:`cifar100`: Extended version with 100 classes
- :doc:`cifar100_c`: Corrupted version of CIFAR-100 for robustness evaluation

References
----------

- Paper: Benchmarking Neural Network Robustness to Common Corruptions and Perturbations (ICLR 2019)
- Zenodo: https://zenodo.org/records/2535967
- License: CC BY 4.0

Citation
--------

.. code-block:: bibtex

    @article{hendrycks2019robustness,
      title={Benchmarking Neural Network Robustness to Common Corruptions and Perturbations},
      author={Dan Hendrycks and Thomas Dietterich},
      journal={Proceedings of the International Conference on Learning Representations},
      year={2019}
    }
