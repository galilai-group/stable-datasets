MNIST-M
=======

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Image%20Classification-blue" alt="Task: Image Classification">
   <img src="https://img.shields.io/badge/Classes-10-green" alt="Classes: 10">
   <img src="https://img.shields.io/badge/Size-32x32x3-orange" alt="Image Size: 32x32x3">
   </p>

Overview
--------
MNIST-M was introduced by Ganin & Lempitsky (2015) as the target domain for unsupervised domain adaptation from MNIST. Each image is an MNIST digit blended over a random color patch cropped from the BSDS500 photo collection, using the absolute difference between the digit and the patch. The result keeps MNIST's labels but has natural-image color and texture statistics, so models trained only on MNIST transfer poorly to it (57.5% accuracy in the original paper, against 98.9% for a model trained on MNIST-M itself).

- **Train**: 59001 images
- **Test**: 9001 images

This builder uses the original release by the authors, linked from Yaroslav Ganin's homepage. Other copies of MNIST-M circulating online (for example the 60000/10000-image, 28x28 versions regenerated with ``keras_mnistm``) are different data, so their sizes and results do not match this release.

.. image:: teasers/mnist_m_teaser.png
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
     - Digit label (0-9)

Usage Example
-------------

**Basic Usage**

.. code-block:: python

    from stable_datasets.images.mnist_m import MNISTM

    # First run will download + prepare cache, then return the split
    ds = MNISTM(split="train")

    # If you omit the split (split=None), you get a StableDatasetDict with all available splits
    ds_all = MNISTM(split=None)

    sample = ds[0]
    print(sample.keys())  # {"image", "label"}

    # Optional: make it PyTorch-friendly
    ds_torch = ds.with_format("torch")

References
----------

- Official website: https://yaroslav.ganin.net/ (the "MNIST-M" link under *Unsupervised Domain Adaptation by Backpropagation*)
- Download: ``mnist_m.tar.gz`` from Google Drive (MD5 ``859df31c91afe82e80e5012ba928f279``; no checksum is published with the release)
- License: Unknown. The release declares no license. The data is derived from MNIST and BSDS500.

Citation
--------

.. code-block:: bibtex

    @inproceedings{ganin2015unsupervised,
        title     = {Unsupervised Domain Adaptation by Backpropagation},
        author    = {Ganin, Yaroslav and Lempitsky, Victor},
        booktitle = {Proceedings of the 32nd International Conference on Machine Learning},
        pages     = {1180--1189},
        year      = {2015}}

    @article{ganin2016domain,
        title     = {Domain-Adversarial Training of Neural Networks},
        author    = {Ganin, Yaroslav and Ustinova, Evgeniya and Ajakan, Hana and Germain, Pascal and Larochelle, Hugo and Laviolette, Fran{\c{c}}ois and Marchand, Mario and Lempitsky, Victor},
        journal   = {Journal of Machine Learning Research},
        volume    = {17},
        number    = {59},
        pages     = {1--35},
        year      = {2016}}
