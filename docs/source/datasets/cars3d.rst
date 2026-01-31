Cars3D
======

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Disentangled%20Representation-blue" alt="Task: Disentangled Representation Learning">
   <img src="https://img.shields.io/badge/Factors-3-green" alt="Latent Factors: 3">
   <img src="https://img.shields.io/badge/Size-128x128-orange" alt="Image Size: 128x128x3">
   </p>

Overview
--------

The Cars3D dataset is a synthetic benchmark widely used for **disentangled and unsupervised representation learning**.  
It consists of rendered images of 3D car models under controlled variations of viewpoint factors, with all generative factors fully known.

Each image depicts a single car model rendered from a specific **azimuth** and **elevation** angle. The dataset forms a **complete Cartesian product** of car identity and viewpoint factors, making it suitable for studying factor disentanglement, invariance, and controllable representation learning.

- **Total images**: 
- **Image resolution**: 128×128×3 (RGB)

Latent Factors of Variation
--------------------------

The dataset is generated from three independent latent factors:

.. list-table::
   :header-rows: 1
   :widths: 20 40

   * - Factor
     - Discrete Values
   * - ``car_type``
     - {0, ..., 182} (183 distinct car models)
   * - ``azimuth``
     - {0, ..., 23} (viewpoint rotation around the vertical axis)
   * - ``elevation``
     - {0, ..., 3} (camera elevation angle)

Each image corresponds to a **unique combination** of these factors.

.. image:: teasers/cars3d_teaser.gif
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
     - 128×128 RGB image
   * - ``car_type``
     - ``int``
     - Index of the car model (0–182)
   * - ``elevation``
     - ``int``
     - Elevation index (0–3)
   * - ``azimuth``
     - ``int``
     - Azimuth index (0–23)
   * - ``label``
     - ``List[int]``
     - Discrete latent indices: ``[car_type, elevation, azimuth]``

Usage Example
-------------

**Basic Usage**

.. code-block:: python

    from stable_datasets.images.cars3d import CARS3D

    ds = CARS3D(split="train")

    sample = ds[0]
    print(sample.keys())

    image = sample["image"]
    label = sample["label"]

    car_type, elevation, azimuth = label

    ds_torch = ds.with_format("torch")

Why No Train/Test Split?
-----------------------

Cars3D does not provide an official train/test split.  
It is designed for **representation learning and disentanglement research**, where evaluation typically focuses on:

- Factor-wise predictability
- Disentanglement metrics
- Controlled generalization across latent variables

Since the dataset enumerates all combinations of car identity and viewpoints, common protocols rely on post-hoc evaluation rather than predefined semantic splits.

Related Datasets
----------------

- **Shapes3D**: 3D shapes with color, scale, and orientation factors
- **MPI3D**: Real and simulated robotic scenes with known generative factors
- **dSprites**: 2D shape-based disentanglement benchmark

References
----------

- Dataset source: http://www.scottreed.info/
- Original release: Reed et al., NIPS 2015 (analogy dataset)
- Common usage: Locatello et al., *Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations*, ICML 2019
- License: Apache License 2.0

Citation
--------

.. code-block:: bibtex

    @inproceedings{locatello2019challenging,
      title={Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations},
      author={Locatello, Francesco and Bauer, Stefan and Lucic, Mario and
              Raetsch, Gunnar and Gelly, Sylvain and Sch{\"o}lkopf, Bernhard and Bachem, Olivier},
      booktitle={International Conference on Machine Learning},
      pages={4114--4124},
      year={2019}
    }
