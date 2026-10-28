AID Scene Classification
========================

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Scene%20Classification-blue" alt="Task: Scene Classification">
   <img src="https://img.shields.io/badge/Classes-30-green" alt="Classes: 30">
   <img src="https://img.shields.io/badge/Domain-Remote%20Sensing-orange" alt="Domain: Remote Sensing">
   </p>

Overview
--------

AID (Aerial Image Dataset) is a remote sensing scene classification dataset with 30 scene categories such as airport, port, forest, and residential areas.

This builder downloads the archive from Kaggle and exposes it as a single split:

- **all**: all samples across all 30 classes

Data Structure
--------------

When accessing an example with ``ds[i]``, the sample contains:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Key
     - Type
     - Description
   * - ``image``
     - ``PIL.Image.Image``
     - RGB aerial scene image
   * - ``label``
     - int
     - Scene class index (0-29)

Usage Example
-------------

.. code-block:: python

    from stable_datasets.images.aid_scene import AIDScene

    ds = AIDScene(split="all")

    sample = ds[0]
    print(sample.keys())  # {"image", "label"}
    print(ds.info.features["label"].names)

Kaggle Requirements
-------------------

This dataset is downloaded with the **Kaggle CLI**. Install it and configure an API key:

.. code-block:: bash

    pip install kaggle

**Option A — token file (recommended)**

.. code-block:: bash

    mkdir -p ~/.kaggle
    mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json

Create ``kaggle.json`` from Kaggle: **Account → API → Create New Token**.

**Option B — environment variables (current shell only)**

.. code-block:: bash

    export KAGGLE_USERNAME="<your_kaggle_username>"
    export KAGGLE_KEY="<your_api_key>"

References
----------

- Kaggle dataset: https://www.kaggle.com/datasets/jiayuanchengala/aid-scene-classification-datasets

Citation
--------

.. code-block:: bibtex

    @article{xia2017aid,
            title={AID: A benchmark data set for performance evaluation of aerial scene classification},
            author={Xia, Gui-Song and Hu, Jingwen and Hu, Fan and Shi, Baoguang and Bai, Xiang and Zhong, Yanfei and Zhang, Liangpei and Lu, Xiaoqiang},
            journal={IEEE Transactions on Geoscience and Remote Sensing},
            volume={55},
            number={7},
            pages={3965--3981},
            year={2017},
            publisher={IEEE}
            }
