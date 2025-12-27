EMNIST
======

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Image%20Classification-blue" alt="Task: Image Classification">
   <img src="https://img.shields.io/badge/Classes-10--62-green" alt="Classes: 10-62">
   <img src="https://img.shields.io/badge/Size-28x28-orange" alt="Image Size: 28x28">
   </p>

Overview
--------

The EMNIST (Extended MNIST) dataset is a set of handwritten character digits and letters derived from the NIST Special Database 19. It is designed to follow the exact same file structure and processing as the original MNIST dataset, serving as a drop-in replacement that is significantly harder due to the inclusion of letters.

The dataset includes **6 different splits (configurations)**. The number of images varies by configuration:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Config
     - Classes
     - Train Images
     - Test Images
     - Total
   * - **digits**
     - 10
     - 240,000
     - 40,000
     - 280,000
   * - **letters**
     - 26
     - 124,800
     - 20,800
     - 145,600
   * - **mnist**
     - 10
     - 60,000
     - 10,000
     - 70,000
   * - **balanced**
     - 47
     - 112,800
     - 18,800
     - 131,600
   * - **byclass**
     - 62
     - 697,932
     - 116,323
     - 814,255
   * - **bymerge**
     - 47
     - 697,932
     - 116,323
     - 814,255

All images are 28×28 grayscale.

.. image:: teasers/e_mnist_teaser.png
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
   * - image
     - PIL.Image.Image
     - 28×28 grayscale handwritten character image
   * - label
     - int
     - Class label (Range depends on the selected config)

Usage Example
-------------

Basic Usage
~~~~~~~~~~~

You must specify a ``config_name`` when loading EMNIST to choose the split (e.g., ``"digits"``, ``"letters"``, ``"balanced"``).

.. code-block:: python

    from stable_datasets.images.e_mnist import EMNIST

    # Load the 'digits' variant (0-9)
    ds_train = EMNIST(config_name="digits", split="train")
    ds_test = EMNIST(config_name="digits", split="test")
    
    # Load the 'letters' variant (A-Z)
    ds_letters = EMNIST(config_name="letters", split="train")

    sample = ds_train[0]
    print(sample.keys())  # {"image", "label"}

    # Optional: make it PyTorch-friendly
    ds_train_torch = ds_train.with_format("torch")

References
----------

Official website: https://www.nist.gov/itl/iad/image-group/emnist-dataset

Citation
--------

.. code-block:: bibtex

    @misc{cohen2017emnistextensionmnisthandwritten,
      title={EMNIST: an extension of MNIST to handwritten letters},
      author={Gregory Cohen and Saeed Afshar and Jonathan Tapson and André van Schaik},
      year={2017},
      eprint={1702.05373},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1702.05373},
    }