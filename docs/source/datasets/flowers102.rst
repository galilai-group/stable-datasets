Flowers102
==========

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Image%20Classification-blue" alt="Task: Image Classification">
   <img src="https://img.shields.io/badge/Classes-102-green" alt="Classes: 102">
   <img src="https://img.shields.io/badge/Size-Variable-orange" alt="Image Size: Variable">
   </p>

Overview
--------

The Flowers102 dataset is a fine-grained image classification benchmark consisting of 8,189 images across 102 flower categories commonly found in the United Kingdom. Unlike standard datasets, the test set is significantly larger than the training set, and images vary in scale, pose, and light.

- **Train**: 1,020 images (10 per class)
- **Validation**: 1,020 images (10 per class)
- **Test**: 6,149 images (variable per class, minimum 20)

.. image:: teasers/flowers102_teaser.png
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
     - ``PIL.Image.Image``
     - Variable resolution RGB flower image
   * - ``label``
     - int
     - Class label (0-101)

Usage Example
-------------

**Basic Usage**

.. code-block:: python

    from stable_datasets.images.flowers102 import Flowers102

    # First run will download + prepare cache, then return the split as a HF Dataset
    ds_train = Flowers102(split="train")
    ds_valid = Flowers102(split="validation")
    ds_test = Flowers102(split="test")

    # If you omit the split (split=None), you get a DatasetDict with all available splits
    ds_all = Flowers102(split=None)

    sample = ds_train[0]
    print(sample.keys())  # {"image", "label"}
    print(f"Label: {sample['label']}") # e.g., 0 (Pink Primrose)

    # Optional: make it PyTorch-friendly
    ds_train_torch = ds_train.with_format("torch")
    ds_test_torch = ds_test.with_format("torch")

References
----------

- Official website: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

Citation
--------

.. code-block:: bibtex

    @inproceedings{nilsback2008flowers102,
      title={Automated flower classification over a large number of classes},
      author={Nilsback, Maria-Elena and Zisserman, Andrew},
      booktitle={2008 Sixth Indian conference on computer vision, graphics \& image processing},
      pages={722--729},
      year={2008},
      organization={IEEE}
    }