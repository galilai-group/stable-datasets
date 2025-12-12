CIFAR-10
========

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Image%20Classification-blue" alt="Task: Image Classification">
   <img src="https://img.shields.io/badge/Classes-10-green" alt="Classes: 10">
   <img src="https://img.shields.io/badge/Size-32x32-orange" alt="Image Size: 32x32">
   </p>

Overview
--------

The CIFAR-10 dataset is a widely-used image classification benchmark for single-label classification consisting of 60,000 32×32 RGB color images across 10 balanced classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck), with 6,000 images per class.

- **Train**: 50,000 images (5,000 per class)
- **Test**: 10,000 images (1,000 per class)

Data Structure
--------------

When accessing an example using ``dataset['train'][i]``, you will receive a dictionary with the following keys:

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
     - Class label (0-9)

Usage Example
-------------

**Basic Usage**

.. code-block:: python

    from stable_datasets import load_dataset

    # Load the dataset
    dataset = load_dataset('cifar10')
    
    # Access a single example
    example = dataset['train'][0]
    image = example['image']  # 32x32x3 RGB image
    label = example['label']  # integer from 0-9
    
    # Iterate over the dataset
    for item in dataset['train']:
        image, label = item['image'], item['label']
        # Your processing here

Related Datasets
----------------

- **CIFAR-100**: Extended version with 100 classes, sharing the same image dimensions
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

