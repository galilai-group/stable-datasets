Fashion-MNIST
=============

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Image%20Classification-blue" alt="Task: Image Classification">
   <img src="https://img.shields.io/badge/Classes-10-green" alt="Classes: 10">
   <img src="https://img.shields.io/badge/Size-28x28-orange" alt="Image Size: 28x28">
   </p>

Overview
--------

The Fashion-MNIST dataset is a widely-used image classification benchmark for single-label classification consisting of 70,000 28×28 grayscale images across 10 balanced classes (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot), with 7,000 images per class.

- **Train**: 60,000 images (6,000 per class)
- **Test**: 10,000 images (1,000 per class)

.. image:: teasers/fashion_mnist_teaser.png
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

    from stable_datasets.images.fashion_mnist import FashionMNIST

    # First run will download + prepare cache, then return the split as a HF Dataset
    ds = FashionMNIST(split="train")

    # If you omit the split (split=None), you get a DatasetDict with all available splits
    ds_all = FashionMNIST(split=None)

    sample = ds[0]
    print(sample.keys())  # {"image", "label"}

    # Optional: make it PyTorch-friendly
    ds_torch = ds.with_format("torch")

Related Datasets
----------------

- **MNIST**: Original handwritten digit dataset with the same dimensions
- **KMNIST**: Kuzushiji-MNIST, a drop-in replacement with Japanese characters
- **EMNIST**: Extended MNIST with letters and digits

References
----------

- Official website: https://github.com/zalandoresearch/fashion-mnist
- License: MIT License

Citation
--------

.. code-block:: bibtex

    @article{xiao2017fashion,
    title={Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms},
    author={Xiao, Han and Rasul, Kashif and Vollgraf, Roland},
    journal={arXiv preprint arXiv:1708.07747},
    year={2017}
    }
