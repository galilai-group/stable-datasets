Arabic Characters
=================

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Image%20Classification-blue" alt="Task: Image Classification">
   <img src="https://img.shields.io/badge/Classes-28-green" alt="Classes: 28">
   <img src="https://img.shields.io/badge/Size-32x32-orange" alt="Image Size: 32x32">
   </p>

Overview
--------

The Arabic Handwritten Characters dataset is a single-label image classification benchmark consisting of 16,800 handwritten character images across 28 classes (from *alef* to *yeh*). Images are provided as 32×32 PNGs and split into mutually-exclusive writers for train and test.

- **Train**: 13,440 images (480 per class)
- **Test**: 3,360 images (120 per class)

.. image:: teasers/arabic_characters_teaser.png
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
     - 32×32 handwritten character image
   * - ``label``
     - int
     - Class label (0-27)

Usage Example
-------------

**Basic Usage**

.. code-block:: python

    from stable_datasets.images.arabic_characters import ArabicCharacters

    # First run will download + prepare cache, then return the split as a HF Dataset
    ds_train = ArabicCharacters(split="train")
    ds_test = ArabicCharacters(split="test")

    # If you omit the split (split=None), you get a DatasetDict with all available splits
    ds_all = ArabicCharacters(split=None)

    sample = ds_train[0]
    print(sample.keys())  # {"image", "label"}

    # Optional: make it PyTorch-friendly
    ds_train_torch = ds_train.with_format("torch")
    ds_test_torch = ds_test.with_format("torch")

References
----------

- Official website: https://github.com/mloey/Arabic-Handwritten-Characters-Dataset

Citation
--------

.. code-block:: bibtex

    @article{el2017arabic,
      title={Arabic handwritten characters recognition using convolutional neural network},
      author={El-Sawy, Ahmed and Loey, Mohamed and El-Bakry, Hazem},
      journal={WSEAS Transactions on Computer Research},
      volume={5},
      pages={11--19},
      year={2017}
    }


