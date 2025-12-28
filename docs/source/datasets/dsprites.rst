dSprites
========

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Disentangled%20Representation-blue" alt="Task: Disentangled Representation Learning">
   <img src="https://img.shields.io/badge/Factors-6-green" alt="Latent Factors: 6">
   <img src="https://img.shields.io/badge/Size-64x64-orange" alt="Image Size: 64x64">
   </p>

Overview
--------

The dSprites dataset is a synthetic benchmark designed for **disentangled and unsupervised representation learning**. It consists of procedurally generated **binary black-and-white images** of simple 2D shapes, rendered under controlled and fully known generative factors.

The dataset contains **all possible combinations** of six latent factors of variation, with each combination appearing exactly once. This complete Cartesian product structure makes dSprites a standard benchmark for evaluating disentanglement, factor predictability, and interpretability of learned representations.

- **Total images**: 737,280
- **Image resolution**: 64×64 (binary)

Latent Factors of Variation
--------------------------

The dataset is generated from six independent latent factors:

.. list-table::
   :header-rows: 1
   :widths: 20 30 30

   * - Factor
     - Discrete Values
     - Continuous Values
   * - ``color``
     - {0}
     - 1.0 (fixed, white)
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

Each image corresponds to a **unique combination** of these factors.

.. image:: teasers/dsprites_teaser.gif
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
     - 64×64 binary image
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

Usage Example
-------------

**Basic Usage**

.. code-block:: python

    from stable_datasets.images.dsprites import DSprites

    # First run will download + prepare cache, then return the split as a HF Dataset
    ds = DSprites(split="train")

    # If you omit the split (split=None), you get a DatasetDict with all available splits
    ds_all = DSprites(split=None)

    sample = ds[0]
    print(sample.keys())

    image = sample["image"]
    factors = sample["label"]
    factor_values = sample["label_values"]

    # Optional: make it PyTorch-friendly
    ds_torch = ds.with_format("torch")


Why No Train/Test Split?
-----------------------

The dSprites dataset does not define an official train/test split.  
It is intended for **representation learning research**, where models are trained to capture underlying factors of variation rather than to generalize across semantic classes.

Because the dataset is a complete Cartesian product of all factor combinations, common evaluation protocols rely on:

- Factor-wise generalization
- Metric-based disentanglement scores
- Controlled interventions on latent variables

Related Datasets
----------------

- **dSprites-Color**: Colored variant of dSprites
- **dSprites-Noisy**: Noisy background variant
- **dSprites-Scream**: Backgrounds replaced with natural images

References
----------

- Dataset repository: https://github.com/google-deepmind/dsprites-dataset
- License: zlib/libpng License
- Paper: Higgins et al., *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*, ICLR 2017

Citation
--------

.. code-block:: bibtex

    @inproceedings{higgins2017beta,
      title={beta-vae: Learning basic visual concepts with a constrained variational framework},
      author={Higgins, Irina and Matthey, Loic and Pal, Arka and Burgess, Christopher and
              Glorot, Xavier and Botvinick, Matthew and Mohamed, Shakir and Lerchner, Alexander},
      booktitle={International Conference on Learning Representations},
      year={2017}
    }
