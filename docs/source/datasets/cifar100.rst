CIFAR-100
=========

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Image%20Classification-blue" alt="Task: Image Classification">
   <img src="https://img.shields.io/badge/Classes-100-green" alt="Classes: 100">
   <img src="https://img.shields.io/badge/Size-32x32-orange" alt="Image Size: 32x32">
   </p>

Overview
--------

The CIFAR-100 dataset is an extended version of CIFAR-10, consisting of 60,000 32×32 RGB color images across 100 fine-grained classes, grouped into 20 superclasses. Each fine class contains 600 images, and each superclass contains 3,000 images.

- **Train**: 50,000 images (500 per class)
- **Test**: 10,000 images (100 per class)

.. image:: teasers/cifar100_teaser.png
   :align: center
   :width: 90%

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
     - PIL Image / numpy array
     - 32×32×3 RGB image
   * - ``label``
     - int
     - Fine-grained class label (0-99)
   * - ``superclass``
     - int
     - Coarse superclass label (0-19)

Usage Example
-------------

**Basic Usage**

.. code-block:: python

    from stable_datasets.images.cifar100 import CIFAR100

    # First run will download + prepare cache, then return the split as a HF Dataset
    ds = CIFAR100(split="train")

    # If you omit the split (split=None), you get a DatasetDict with all available splits
    ds_all = CIFAR100(split=None)

    sample = ds[0]
    print(sample.keys())  # {"image", "label", "superclass"}

    # Optional: make it PyTorch-friendly
    ds_torch = ds.with_format("torch")

Related Datasets
----------------

- :doc:`cifar10`: Simplified version with 10 classes, sharing the same image dimensions
- **CIFAR-10-C**: Corrupted version of CIFAR-10 for testing robustness (available in stable-datasets as ``cifar10_c``)
- **CIFAR-100-C**: Corrupted version of CIFAR-100 (available in stable-datasets as ``cifar100_c``)

References
----------

- Official website: https://www.cs.toronto.edu/~kriz/cifar.html
- License: MIT License

Citation
--------

.. code-block:: bibtex

    @Techreport{krizhevsky2009learning,
    author = {Krizhevsky, Alex and Hinton, Geoffrey},
    address = {Toronto, Ontario},
    institution = {University of Toronto},
    number = {0},
    publisher = {Technical report, University of Toronto},
    title = {Learning multiple layers of features from tiny images},
    year = {2009},
    title_with_no_special_chars = {Learning multiple layers of features from tiny images},
    url = {https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf}
    }
