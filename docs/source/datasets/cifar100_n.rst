CIFAR-100N
==========

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Image%20Classification-blue" alt="Task: Image Classification">
   <img src="https://img.shields.io/badge/Classes-100-green" alt="Classes: 100">
   <img src="https://img.shields.io/badge/Size-32x32-orange" alt="Image Size: 32x32">
   </p>

Overview
--------

CIFAR-100N pairs the 50,000 CIFAR-100 training images with their clean fine and coarse labels and the human noisy labels from the CIFAR-N release. It also provides the 10,000 standard CIFAR-100 test images and labels. The test examples have ``None`` for ``noisy_label`` because CIFAR-N annotations cover only training images.

- **Train**: 50,000 images (500 per class)
- **Test**: 10,000 images (100 per class)

.. image:: teasers/cifar100n_teaser.png
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
     - 32x32x3 RGB image
   * - ``label``
     - int
     - Clean fine-grained class label (0-99)
   * - ``coarse_label``
     - int
     - Clean superclass label (0-19)
   * - ``noisy_label``
     - int or ``None``
     - Human annotation label (0-99); ``None`` on the test split

Usage Example
-------------

**Basic Usage**

.. code-block:: python

    from stable_datasets.images.cifar100_n import CIFAR100N

    train = CIFAR100N(split="train")
    test = CIFAR100N(split="test")
    sample = train[0]

    print(sample["label"], sample["coarse_label"], sample["noisy_label"])
    print(test[0]["noisy_label"])  # None: no human annotations are provided for test images

Related Datasets
----------------

- :doc:`cifar10`: CIFAR-10 with 10 classes
- :doc:`cifar100`: CIFAR-100 with 100 fine-grained classes
- :doc:`cifar10_c`: CIFAR-10-C corruption benchmark
- :doc:`cifar100_c`: CIFAR-100-C corruption benchmark
- :doc:`cifar10_n`: CIFAR-10 with real-world human annotations

References
----------

- CIFAR-N release: https://github.com/UCSC-REAL/cifar-10-100n
- CIFAR-100 dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- Wei et al., *Learning with Noisy Labels Revisited: A Study Using Real-World Human Annotations*, ICLR 2022: https://openreview.net/forum?id=TBWA6PLJZQm

Citation
--------

.. code-block:: bibtex

    @inproceedings{wei2022learning,
      title={Learning with Noisy Labels Revisited: A Study Using Real-World Human Annotations},
      author={Jiaheng Wei and Zhaowei Zhu and Hao Cheng and Tongliang Liu and Gang Niu and Yang Liu},
      booktitle={International Conference on Learning Representations},
      year={2022},
      url={https://openreview.net/forum?id=TBWA6PLJZQm}
    }




