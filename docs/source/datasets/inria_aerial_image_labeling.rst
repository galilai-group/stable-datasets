Inria Aerial Image Labeling
===========================

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Semantic%20Segmentation-blue" alt="Task: Semantic Segmentation">
   <img src="https://img.shields.io/badge/Classes-2%20(building%2Fbackground)-green" alt="Classes: 2">
   <img src="https://img.shields.io/badge/Domain-Remote%20Sensing-orange" alt="Domain: Remote Sensing">
   </p>

Overview
--------

The `Inria Aerial Image Labeling`_ benchmark provides high-resolution aerial orthophotos (0.3 m) with pixel-wise labels for **building** vs. **not building**. Training tiles include public ground truth; the official test set is images only (no public masks), so in this builder the **test** split returns ``mask=None``.

Raw archives are downloaded from the Kaggle mirror `sagar100rathod/inria-aerial-image-labeling-dataset` (see `Kaggle dataset`_). Always cite the original paper and follow the dataset license from the official site.

.. _`Inria Aerial Image Labeling`: https://project.inria.fr/aerialimagelabeling/
.. _`Kaggle dataset`: https://www.kaggle.com/datasets/sagar100rathod/inria-aerial-image-labeling-dataset

Data Structure
--------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Key
     - Type
     - Description
   * - ``image``
     - ``PIL.Image.Image``
     - RGB orthophoto (often large GeoTIFF decoded by Pillow)
   * - ``mask``
     - ``PIL.Image.Image`` or ``None``
     - Training: label mask aligned with the image; test: ``None`` (no public GT)

Splits
------

- **train**: paired ``image`` and ``mask``
- **test**: ``image`` only if a test image folder is present in the archive; ``mask`` is ``None``

Usage Example
-------------

.. code-block:: python

    from stable_datasets.images.inria_aerial_image_labeling import InriaAerialImageLabeling

    train_ds = InriaAerialImageLabeling(split="train")
    sample = train_ds[0]
    print(sample.keys())  # {"image", "mask"}

    # Optional: load all splits as a dict
    ds_all = InriaAerialImageLabeling(split=None)

Kaggle setup
------------

Install the **Kaggle CLI** and configure an **API key** (same pattern as other Kaggle-backed builders):

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

Run the dataset test (downloads a large archive; requires Kaggle auth above):

.. code-block:: bash

    pytest -q -rs stable_datasets/tests/images/test_inria_aerial_image_labeling.py

The test is marked ``large`` because of download size and processing time.

If ``kaggle datasets download`` dies with **SIGKILL** (exit code -9), the process was usually killed by the OS for **low memory** while handling the large zip—not because your API key is wrong. Free RAM, download the archive in a normal terminal first, delete any partial zip, then run the test again.

References
----------

- Official dataset: https://project.inria.fr/aerialimagelabeling/
- Kaggle mirror: https://www.kaggle.com/datasets/sagar100rathod/inria-aerial-image-labeling-dataset

Citation
--------

.. code-block:: bibtex

    @inproceedings{maggiori2017dataset,
      title={Can Semantic Labeling Methods Generalize to Any City? The Inria Aerial Image Labeling Benchmark},
      author={Maggiori, Emmanuel and Tarabalka, Yuliya and Charpiat, Guillaume and Alliez, Pierre},
      booktitle={IEEE International Geoscience and Remote Sensing Symposium (IGARSS)},
      year={2017},
      organization={IEEE}
    }
