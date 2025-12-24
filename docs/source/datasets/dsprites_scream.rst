Scream dSprites
===============

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Disentangled%20Representation-blue" alt="Task: Disentangled Representation Learning">
   <img src="https://img.shields.io/badge/Factors-6-green" alt="Latent Factors: 6">
   <img src="https://img.shields.io/badge/Size-64x64x3-orange" alt="Image Size: 64x64x3">
   </p>

Overview
--------

The Scream dSprites dataset is a synthetic benchmark designed for **disentangled and unsupervised representation learning**. It is a variant of the original dSprites dataset, where each object sprite is embedded onto a **random patch from a Scream painting image** used as background.

In this Scream variant, the object is rendered by **inverting the pixel colors** of the sprite region on top of the background patch. This introduces a **structured and highly textured background**, enabling evaluation of robustness to complex background variation and testing how well models can disentangle latent factors under realistic nuisance conditions.

The dataset contains **all possible combinations** of six latent factors of variation inherited from dSprites, with each combination appearing exactly once.

- **Total images**: 737,280
- **Image resolution**: 64×64×3 (textured background with inverted object)

Latent Factors of Variation
--------------------------

The dataset is generated from six independent latent factors, consistent with the original dSprites specification.  
In this variant, no additional latent factor is introduced; the background texture is applied independently of the ground-truth factors.

.. list-table::
   :header-rows: 1
   :widths: 20 30 30

   * - Factor
     - Discrete Values
     - Description
   * - ``color``
     - {0}
     - Fixed label; object pixels inverted on Scream background
   * - ``shape``
     - {0, 1, 2}
     - Square, ellipse, heart
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
The factor indexing and ordering are identical to the original dSprites dataset.

.. image:: teasers/dsprites_scream_teaser.gif
   :align: center
   :width: 90%

Scream Background Visualization
-------------------------------

The background patches are sampled from a fixed Scream painting image.

.. image:: imgs/scream.png
   :align: center
   :width: 70%

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
     - 64×64×3 image with Scream background
   * - ``label``
     - ``List[int]``
     - Discrete latent indices: ``[color, shape, scale, orientation, posX, posY]``
   * - ``label_values``
     - ``List[float]``
     - Continuous latent values corresponding to ``label``
   * - ``color`` … ``posY``
     - ``int``
     - Individual discrete latent factors
   * - ``colorValue`` … ``posYValue``
     - ``float``
     - Individual continuous latent values

**Note:**  
The ``color`` and ``colorValue`` fields remain fixed to preserve compatibility with the original dSprites format.  
The object is rendered by **pixel inversion** on top of the Scream background.

Usage Example
-------------

**Basic Usage**

.. code-block:: python

    from stable_datasets.images.dsprites_scream import DSpritesScream

    # First run will download + prepare cache, then return the split as a HF Dataset
    ds = DSpritesScream(split="train")

    # If you omit the split (split=None), you get a DatasetDict with all available splits
    ds_all = DSpritesScream(split=None)

    sample = ds[0]
    print(sample.keys())

    image = sample["image"]
    factors = sample["label"]
    factor_values = sample["label_values"]

    # Optional: make it PyTorch-friendly
    ds_torch = ds.with_format("torch")

Why No Train/Test Split?
-----------------------

The Scream dSprites dataset does not define an official train/test split.  
It is intended for **representation learning research**, where models are trained to capture underlying factors of variation rather than to generalize across semantic classes.

Because the dataset is a complete Cartesian product of all factor combinations, common evaluation protocols rely on:

- Factor-wise generalization
- Metric-based disentanglement scores
- Robustness to structured background textures
- Controlled interventions on latent variables

Related Datasets
----------------

- **dSprites**: Original grayscale version
- **dSprites-Color**: Variant with random object color
- **dSprites-Noisy**: Variant with random background noise

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
