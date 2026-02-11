Shapes3D
========

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Disentangled%20Representation-blue" alt="Task: Disentangled Representation Learning">
   <img src="https://img.shields.io/badge/Factors-6-green" alt="Latent Factors: 6">
   <img src="https://img.shields.io/badge/Size-64x64-orange" alt="Image Size: 64x64">
   </p>

Overview
--------

The **Shapes3D dataset** (also known as **3dshapes**) is a synthetic benchmark designed for
**disentangled and unsupervised representation learning**.
It was introduced in the **FactorVAE** paper by **Kim & Mnih (ICML 2018)** as a standard testbed
for learning interpretable latent representations.

The dataset consists of **procedurally generated images of 3D scenes**, where a single object
is rendered in a simple environment with **six independent, ground-truth factors of variation**
that are explicitly controlled.

Unlike real-world datasets, Shapes3D is generated as a **complete Cartesian product** over all
factors. Every possible combination of factor values appears **exactly once**, making the dataset
well suited for systematic and controlled disentanglement evaluation.

- **Total images**: 480,000
- **Image resolution**: 64×64×3 (RGB)

Latent Factors of Variation
--------------------------

Each image is generated from six independent factors:

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Factor
     - Values
     - Cardinality
   * - ``floor_color``
     - Linearly spaced in [0, 1]
     - 10
   * - ``wall_color``
     - Linearly spaced in [0, 1]
     - 10
   * - ``object_color``
     - Linearly spaced in [0, 1]
     - 10
   * - ``scale``
     - Linearly spaced in [0.75, 1.25]
     - 8
   * - ``shape``
     - {0, 1, 2, 3}
     - 4
   * - ``orientation``
     - Linearly spaced in [-30, 30]
     - 15

Each image corresponds to a **unique combination** of these six factors.
The images are stored in **row-major order**, where the fastest-changing factor is
``orientation`` and the slowest-changing factor is ``floor_color``.

.. image:: teasers/shapes3d_teaser.gif
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
     - 64×64 RGB image
   * - ``label``
     - ``List[float]``
     - Continuous factor values:
       ``[floor, wall, object, scale, shape, orientation]``
   * - ``label_index``
     - ``List[int]``
     - Discrete factor indices corresponding to ``label``
   * - ``floor`` … ``orientation``
     - ``float``
     - Individual continuous factor values
   * - ``floor_idx`` … ``orientation_idx``
     - ``int``
     - Individual discrete factor indices

Usage Example
-------------

**Basic Usage**

.. code-block:: python

    from stable_datasets.images.shapes3d import Shapes3D

    ds = Shapes3D(split="train")

    sample = ds[0]

    image = sample["image"]
    label = sample["label"]
    label_index = sample["label_index"]

    floor, wall, obj, scale, shape, orientation = label
    floor_idx, wall_idx, obj_idx, scale_idx, shape_idx, orientation_idx = label_index

    image.show()

    # Optional: make it PyTorch-friendly
    ds_torch = ds.with_format("torch")

Why No Train/Test Split?
-----------------------

Shapes3D does not define an official train/test split.
The dataset is intended for **representation learning**, where the objective is to discover and
disentangle latent generative factors rather than to generalize across semantic classes.

Because Shapes3D is a **complete Cartesian product** of all factor combinations, most evaluation
protocols rely on:

- Factor-wise interventions
- Disentanglement metrics
- Controlled generalization across factors

References
----------

- Dataset homepage: https://github.com/google-deepmind/3dshapes-dataset
- License: Apache License 2.0
- Paper: Kim & Mnih, *Disentangling by Factorising*, ICML 2018

Citation
--------

.. code-block:: bibtex

    @InProceedings{pmlr-v80-kim18b,
      title = {Disentangling by Factorising},
      author = {Kim, Hyunjik and Mnih, Andriy},
      booktitle = {Proceedings of the 35th International Conference on Machine Learning},
      pages = {2649--2658},
      year = {2018},
      editor = {Dy, Jennifer and Krause, Andreas},
      volume = {80},
      series = {Proceedings of Machine Learning Research},
      month = {10--15 Jul},
      publisher = {PMLR}
    }
