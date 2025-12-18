MedMNIST
=======

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Biomedical%20Image%20Classification-blue" alt="Task: Biomedical Image Classification">
   <img src="https://img.shields.io/badge/Variants-2D%20%2B%203D-green" alt="Variants: 2D + 3D">
   <img src="https://img.shields.io/badge/Size-28x28%20%2F%2028x28x28-orange" alt="Image Size: 28x28 / 28x28x28">
   </p>

Overview
--------

MedMNIST is a large-scale MNIST-like collection of standardized biomedical images. In stable-datasets, MedMNIST is exposed via the `MedMNIST` class, but **the actual dataset depends on the selected variant** (passed as ``config_name``, e.g. ``dermamnist``, ``pathmnist``). Each variant provides train/validation/test splits.

All 2D variants are pre-processed to **28×28** images and all 3D variants are pre-processed to **28×28×28** volumes, with corresponding labels.

.. image:: teasers/medmnist_dermamnist_teaser.png
   :align: center
   :width: 90%

Variants (2D)
-------------

2D variants use 28×28 images.

.. list-table::
   :header-rows: 1
   :widths: 18 10 18 18 18 16

   * - Variant
     - Classes
     - Train
     - Validation
     - Test
     - Notes
   * - ``pathmnist``
     - 9
     - 89,996
     - 10,004
     - 7,180
     -
   * - ``chestmnist``
     - 14
     - 78,468
     - 11,219
     - 22,433
     - multi-label
   * - ``dermamnist``
     - 7
     - 7,007
     - 1,003
     - 2,005
     -
   * - ``octmnist``
     - 4
     - 97,477
     - 10,832
     - 1,000
     -
   * - ``pneumoniamnist``
     - 2
     - 4,708
     - 524
     - 624
     -
   * - ``retinamnist``
     - 5
     - 1,080
     - 120
     - 400
     - ordinal regression
   * - ``breastmnist``
     - 2
     - 546
     - 78
     - 156
     -
   * - ``bloodmnist``
     - 8
     - 11,959
     - 1,712
     - 3,421
     -
   * - ``tissuemnist``
     - 8
     - 165,466
     - 23,640
     - 47,280
     -
   * - ``organamnist``
     - 11
     - 34,561
     - 6,491
     - 17,778
     -
   * - ``organcmnist``
     - 11
     - 12,975
     - 2,392
     - 8,216
     -
   * - ``organsmnist``
     - 11
     - 13,932
     - 2,452
     - 8,827
     -

Variants (3D)
-------------

3D variants use 28×28×28 volumes.

.. list-table::
   :header-rows: 1
   :widths: 18 10 18 18 18

   * - Variant
     - Classes
     - Train
     - Validation
     - Test
   * - ``organmnist3d``
     - 11
     - 971
     - 161
     - 610
   * - ``nodulemnist3d``
     - 2
     - 1,158
     - 165
     - 310
   * - ``adrenalmnist3d``
     - 2
     - 1,188
     - 98
     - 298
   * - ``fracturemnist3d``
     - 3
     - 1,027
     - 103
     - 240
   * - ``vesselmnist3d``
     - 2
     - 1,335
     - 191
     - 382
   * - ``synapsemnist3d``
     - 2
     - 1,230
     - 177
     - 352

Data Structure
--------------

When accessing an example using ``ds[i]``, you will receive a dictionary with the following keys.

2D variants
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Key
     - Type
     - Description
   * - ``image``
     - ``PIL.Image.Image``
     - 28×28 image
   * - ``label``
     - int / list[int]
     - Class label (range depends on the selected variant; ``chestmnist`` uses a multi-label vector)

3D variants
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Key
     - Type
     - Description
   * - ``image``
     - list
     - 28×28×28 volume
   * - ``label``
     - int
     - Class label (range depends on the selected variant)

Usage Example
-------------

**Basic Usage (2D variant)**

.. code-block:: python

    from stable_datasets.images.med_mnist import MedMNIST

    # Pick a 2D variant via config_name
    variant = "dermamnist"

    ds_train = MedMNIST(split="train", config_name=variant)
    ds_val = MedMNIST(split="validation", config_name=variant)
    ds_test = MedMNIST(split="test", config_name=variant)

    sample = ds_train[0]
    print(sample.keys())  # {"image", "label"}

    image = sample["image"]  # PIL.Image.Image
    label = sample["label"]  # int

    # Optional: make it PyTorch-friendly
    ds_train_torch = ds_train.with_format("torch")

**Basic Usage (2D multi-label variant: chestmnist)**

.. code-block:: python

    from stable_datasets.images.med_mnist import MedMNIST

    variant = "chestmnist"
    ds_train = MedMNIST(split="train", config_name=variant)

    sample = ds_train[0]
    image = sample["image"]  # PIL.Image.Image
    label = sample["label"]  # multi-label vector (length 14)

    # Example: indices of positive labels
    positives = [i for i, v in enumerate(label) if int(v) == 1]
    print("positive label indices:", positives)

**Basic Usage (3D variant)**

.. code-block:: python

    from stable_datasets.images.med_mnist import MedMNIST

    variant = "organmnist3d"

    ds_train = MedMNIST(split="train", config_name=variant)
    sample = ds_train[0]

    image = sample["image"]  # nested list, shape (28, 28, 28)
    label = sample["label"]  # int

References
----------

- Official website: https://medmnist.com/
- License: CC BY 4.0

Citation
--------

.. code-block:: bibtex

    @article{medmnistv2,
      title={MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image classification},
      author={Yang, Jiancheng and Shi, Rui and Wei, Donglai and Liu, Zequan and Zhao, Lin and Ke, Bilian and Pfister, Hanspeter and Ni, Bingbing},
      journal={Scientific Data},
      volume={10},
      number={1},
      pages={41},
      year={2023},
      publisher={Nature Publishing Group UK London}
    }


