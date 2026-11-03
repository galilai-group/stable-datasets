MNIST-C
=======

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Robustness%20Evaluation-blue" alt="Task: Robustness Evaluation">
   <img src="https://img.shields.io/badge/Classes-10-green" alt="Classes: 10">
   <img src="https://img.shields.io/badge/Size-28x28-orange" alt="Image Size: 28x28">
   <img src="https://img.shields.io/badge/Corruptions-15-red" alt="Corruptions: 15">
   </p>

Overview
--------

MNIST-C is a corrupted version of MNIST designed for benchmarking the out-of-distribution robustness of computer vision models. It applies 15 corruptions (noise, blur, affine transforms, digital and weather-like artifacts) to both the MNIST training and test sets, at a single fixed severity per corruption. Labels and image order are identical across corruptions, so every corrupted image can be paired with its clean original.

Each corruption is a separate builder config, selected with ``config_name``. The ``identity`` config is the uncorrupted MNIST reference shipped in the same archive. Every config has the same splits:

- **Train**: 60,000 images
- **Test**: 10,000 images

All images are 28×28 grayscale. The default config is ``shot_noise``.

.. image:: teasers/mnist_c_shot_noise_teaser.png
   :align: center
   :width: 90%

.. image:: teasers/mnist_c_canny_edges_teaser.png
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
     - 28×28 grayscale corrupted digit image
   * - ``label``
     - int
     - Digit class label (0-9)

Corruption Types
----------------

Configs grouped by corruption type:

**Noise:**

- ``shot_noise``
- ``impulse_noise``

**Blur:**

- ``glass_blur``
- ``motion_blur``

**Affine:**

- ``shear``
- ``scale``
- ``rotate``
- ``translate``

**Digital:**

- ``brightness``
- ``stripe``
- ``dotted_line``
- ``zigzag``
- ``canny_edges``

**Weather:**

- ``fog``
- ``spatter``

**Reference:**

- ``identity`` (clean MNIST, no corruption)

Usage Example
-------------

**Basic Usage**

.. code-block:: python

    from stable_datasets.images.mnist_c import MNISTC

    # Load one corruption (the archive is downloaded once and shared by all configs)
    ds_test = MNISTC(config_name="fog", split="test")

    sample = ds_test[0]
    print(sample.keys())  # {"image", "label"}

    # Optional: make it PyTorch-friendly
    ds_torch = ds_test.with_format("torch")

**Typical Workflow**

.. code-block:: python

    from stable_datasets.images.mnist_c import MNISTC

    # Train on clean MNIST
    train_data = MNISTC(config_name="identity", split="train")
    # ... train your model ...

    # Evaluate robustness on each corruption
    for config in MNISTC.BUILDER_CONFIGS:
        test_data = MNISTC(config_name=config.name, split="test")
        # ... evaluate on corrupted images ...

Related Datasets
----------------

- :doc:`cifar10_c`: Corrupted version of CIFAR-10 for robustness evaluation
- :doc:`tiny_imagenet_c`: Corrupted version of Tiny ImageNet for robustness evaluation
- :doc:`e_mnist`: Extended MNIST with handwritten letters and digits

References
----------

- Paper: MNIST-C: A Robustness Benchmark for Computer Vision (https://arxiv.org/abs/1906.02337)
- Code: https://github.com/google-research/mnist-c
- Zenodo: https://zenodo.org/records/3239543
- License: CC BY 4.0

Citation
--------

.. code-block:: bibtex

    @article{mu2019mnist,
      title={MNIST-C: A Robustness Benchmark for Computer Vision},
      author={Mu, Norman and Gilmer, Justin},
      journal={arXiv preprint arXiv:1906.02337},
      year={2019}
    }
