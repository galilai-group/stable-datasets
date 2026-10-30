Caltech-256
===========

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Image%20Classification-blue" alt="Task: Image Classification">
   <img src="https://img.shields.io/badge/Classes-257-green" alt="Classes: 257">
   <img src="https://img.shields.io/badge/Size-HxWx3-orange" alt="Image Size: HxWx3">
   </p>

Overview
--------

The Caltech-256 dataset is a large-scale image classification benchmark for single-label object recognition.
It contains **257 object categories** with **at least 80 images per class**, resulting in **30,607 images** in total.
The dataset contains a wide variety of objects and substantial variation in viewpoint, scale, lighting, and background.
It does not provide an official train/test split, so this is left for the user to customize.

* **Total**: 30,607 images across 257 classes
* **Classes**: 257 object categories

Images have varying resolutions and aspect ratios. The dataset contains 30,185 RGB and 422 grayscale images, however
all images have been converted to 3-channel RGB format.


.. image:: teasers/caltech256_teaser.png
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
     - RGB image with maximum side length ≤ 7913 pixels
   * - ``label``
     - int
     - Class label in the range ``[0, 256]``

Usage Example
-------------

**Basic Usage**

.. code-block:: python

    from stable_datasets.images.caltech256 import Caltech256

    # Load training split (the only available split)
    ds = Caltech256(split="train")

    sample = ds[0]
    print(sample.keys())  # {"image", "label"}

    # Optional: make it PyTorch-friendly
    ds_torch = ds.with_format("torch")

References
----------

- Official website: https://data.caltech.edu/records/nyy15-4j048
- License: Creative Commons Attribution 4.0 International

Citation
--------

.. code-block:: bibtex

    @article{griffin_holub_perona_2007,
        title = {Caltech-256 Object Category Dataset},
        publisher = {California Institute of Technology},
        author = {Griffin, Gregory and Holub, Alex and Perona, Pietro},
        year = {2007},
        month = {mar}
    }
