The Street View House Numbers Dataset (SVHN)
========

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Image%20Classification-blue" alt="Task: Image Classification">
   <img src="https://img.shields.io/badge/Classes-10-green" alt="Classes: 10">
   <img src="https://img.shields.io/badge/Size-32x32-orange" alt="Image Size: 32x32">
   </p>

Overview
--------

SVHN is a real-world image dataset for developing machine learning and object recognition algorithms with minimal requirement on data preprocessing and formatting. It can be seen as similar in flavor to MNIST (e.g., the images are of small cropped digits), but incorporates an order of magnitude more labeled data (over 600,000 digit images) and comes from a significantly harder, unsolved, real world problem (recognizing digits and numbers in natural scene images). SVHN is obtained from house numbers in Google Street View images.

- **Train**: 73,257 images
- **Test**: 26,032 images
- **Extra**: 531,131 images (additionalally less difficult samples, using as extra training data)

.. image:: teasers/svhn_teaser.png
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
     - 32×32×3 RGB image
   * - ``label``
     - int
     - Class label (0-9)

Usage Example
-------------

**Basic Usage**

.. code-block:: python

    from stable_datasets.images.svhn import SVHN

    # First run will download + prepare cache, then return the split as a HF Dataset
    ds = SVHN(split="train")

    # If you omit the split (split=None), you get a DatasetDict with all available splits
    ds_all = SVHN(split=None)

    sample = ds[0]
    print(sample.keys())  # {"image", "label"}

    # Optional: make it PyTorch-friendly
    ds_torch = ds.with_format("torch")

References
----------

- Official website: http://ufldl.stanford.edu/housenumbers/
- License: MIT License

Citation
--------

.. code-block:: bibtex

    @inproceedings{netzer2011reading,
    title={Reading digits in natural images with unsupervised feature learning},
    author={Netzer, Yuval and Wang, Tao and Coates, Adam and Bissacco, Alessandro and Wu, Baolin and Ng, Andrew Y and others},
    booktitle={NIPS workshop on deep learning and unsupervised feature learning},
    volume={2011},
    number={2},
    pages={4},
    year={2011},
    organization={Granada}

