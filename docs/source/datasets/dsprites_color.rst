Color dSprites
==============

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Disentangled%20Representation-blue" alt="Task: Disentangled Representation Learning">
   <img src="https://img.shields.io/badge/Factors-6-green" alt="Latent Factors: 6">
   <img src="https://img.shields.io/badge/Size-64x64x3-orange" alt="Image Size: 64x64x3">
   </p>

Overview
--------

The Color dSprites dataset is a synthetic benchmark designed for **disentangled and unsupervised representation learning**. It is a variant of the original dSprites dataset, where the object in each image is rendered with a **random RGB color**, while the background remains black.

Compared to the original grayscale dSprites, this variant introduces **random object color per sample**, enabling evaluation of **robustness to color variation** and analysis of how disentanglement methods behave under color transformations that are not part of the ground-truth generative factors.

The dataset contains **all possible combinations** of six latent factors of variation inherited from dSprites, with each combination appearing exactly once.

- **Total images**: 737,280
- **Image resolution**: 64×64x3 (RGB)

Latent Factors of Variation
--------------------------

The dataset is generated from six independent latent factors, consistent with the original dSprites specification.  
In this color variant, the ``color`` factor remains fixed in the labels, while the **actual applied RGB color** is provided separately.

.. list-table::
   :header-rows: 1
   :widths: 20 30 30

   * - Factor
     - Discrete Values
     - Continuous / Actual Values
   * - ``color``
     - {0}
     - Fixed label; actual RGB stored in ``colorRGB`` ∈ [0.5, 1.0]
   * - ``shape``
     - {0, 1, 2}
     - {1.0, 2.0, 3.0} (square, ellipse, heart)
   * - ``scale``
     - {0, ..., 5}
     - Linearly spaced in [0.5, 1.0]
   * - ``orientation``
     - {0, ..., 39}
     - Uniform in [0, 2π] radians
   * - ``posX``
     - {0, ..., 31}
     - Normalized position in [0, 1]
   * - ``posY``
     - {0, ..., 31}
     - Normalized position in [0, 1]

Each image corresponds to a **unique combination** of these six latent factors.  
The random object color does **not** alter the factor indexing order.

.. image:: teasers/dsprites_color_teaser.gif
   :align: center
   :width: 90%

Data Structure
--------------

When accessing an example using ``ds[i]``, you will receive a dictionary with the following keys:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Key
     - Type
     - Description
   * - ``image``
     - ``PIL.Image.Image``
     - 64×64×3 RGB image (black background)
   * - ``label``
     - ``List[int]``
     - Discrete latent indices: ``[color, shape, scale, orientation, posX, posY]``
   * - ``label_values``
     - ``List[float]``
     - Continuous latent values corresponding to ``label``
   * - ``colorRGB``
     - ``List[float]``
     - Actual RGB color applied to the object
   * - ``color`` … ``posY``
     - ``int``
     - Individual discrete latent factors
   * - ``colorValue`` … ``posYValue``
     - ``float``
     - Individual continuous latent values

**Note:**  
The ``color`` and ``colorValue`` fields remain fixed (0 / 1.0) to preserve compatibility with the original dSprites format.  
The actual object color is provided exclusively via ``colorRGB``.

Usage Example
-------------

**Basic Usage**

.. code-block:: python

    from stable_datasets.images.dsprites_color import DSpritesColor

    # First run will download + prepare cache, then return the split as a HF Dataset
    ds = DSpritesColor(split="train")

    # If you omit the split (split=None), you get a DatasetDict with all available splits
    ds_all = DSpritesColor(split=None)

    sample = ds[0]
    print(sample.keys())

    image = sample["image"]
    factors = sample["label"]
    factor_values = sample["label_values"]
    color_rgb = sample["colorRGB"]

    # Optional: make it PyTorch-friendly
    ds_torch = ds.with_format("torch")

Why No Train/Test Split?
-----------------------

The Color dSprites dataset does not define an official train/test split.  
It is intended for **representation learning research**, where models are trained to capture underlying factors of variation rather than to generalize across semantic classes.

Because the dataset is a complete Cartesian product of all factor combinations, common evaluation protocols rely on:

- Factor-wise generalization
- Metric-based disentanglement scores
- Robustness analysis under nuisance variations (e.g., color)
- Controlled interventions on latent variables

Related Datasets
----------------

- **dSprites**: Original grayscale version
- **dSprites-Noisy**: Variant with background noise
- **dSprites-Scream**: Variant with natural image backgrounds

References
----------

- Dataset repository: https://github.com/google-research/disentanglement_lib/
- License: Apache License 2.0
- Paper: Locatello et al., *Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations*, ICML 2019

Citation
--------

.. code-block:: bibtex

    @inproceedings{locatello2019challenging,
      title={Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations},
      author={Locatello, Francesco and Bauer, Stefan and Lucic, Mario and
              Raetsch, Gunnar and Gelly, Sylvain and
              Sch{\"o}lkopf, Bernhard and Bachem, Olivier},
      booktitle={International Conference on Machine Learning},
      year={2019}
    }
