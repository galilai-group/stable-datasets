Kuzushiji-MNIST
===============

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Image%20Classification-blue" alt="Task: Image Classification">
   <img src="https://img.shields.io/badge/Classes-10-green" alt="Classes: 10">
   <img src="https://img.shields.io/badge/Size-28x28-orange" alt="Image Size: 28x28">
   </p>

Overview
--------

The Kuzushiji-MNIST dataset is a widely-used image classification benchmark for single-label classification consisting of 70,000 28×28 grayscale images of cursive Japanese characters across 10 balanced classes (o, ki, su, tsu, na, ha, ma, ya, re, wo), with 7,000 images per class.

- **Train**: 60,000 images (6,000 per class)
- **Test**: 10,000 images (1,000 per class)

.. image:: teasers/k_mnist_teaser.png
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
     - 28×28 grayscale image
   * - ``label``
     - int
     - Class label (0-9)

Usage Example
-------------

**Basic Usage**

.. code-block:: python

    from stable_datasets.images.k_mnist import KMNIST

    # First run will download + prepare cache, then return the split as a HF Dataset
    ds = KMNIST(split="train")

    # If you omit the split (split=None), you get a DatasetDict with all available splits
    ds_all = KMNIST(split=None)

    sample = ds[0]
    print(sample.keys())  # {"image", "label"}

    # Optional: make it PyTorch-friendly
    ds_torch = ds.with_format("torch")

Related Datasets
----------------

- **MNIST**: Original handwritten digit recognition dataset
- **Fashion-MNIST**: Drop-in replacement for MNIST with fashion items
- **EMNIST**: Extended MNIST with letters and digits
- **Kuzushiji-49**: Extended version with 49 classes of Kuzushiji characters

References
----------

- Official website: http://codh.rois.ac.jp/kmnist/
- License: CC BY-SA 4.0

Citation
--------

.. code-block:: bibtex

    @online{clanuwat2018deep,
    author       = {Tarin Clanuwat and Mikel Bober-Irizar and Asanobu Kitamoto and Alex Lamb and Kazuaki Yamamoto and David Ha},
    title        = {Deep Learning for Classical Japanese Literature},
    date         = {2018-12-03},
    year         = {2018},
    eprintclass  = {cs.CV},
    eprinttype   = {arXiv},
    eprint       = {cs.CV/1812.01718}
    }
