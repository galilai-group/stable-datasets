Country-211
========

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Image%20Classification-blue" alt="Task: Image Classification">
   <img src="https://img.shields.io/badge/Classes-211-green" alt="Classes: 211">
   <img src="https://img.shields.io/badge/Size-HxWx3-orange" alt="Image Size: HxWx3">
   </p>

Overview
--------

The Country211 dataset is designed for country classification based on images. It was created to evaluate the geolocation capabilities of machine learning models. The dataset is a filtered subset of the YFCC100m dataset, consisting of images that have GPS coordinates corresponding to an ISO-3166 country code. The dataset is balanced, containing 150 training images, 50 validation images, and 100 test images for each of the 211 countries and territories.

- **Train**: 31,650 images (150 per class)
- **Validation**: 10,550 images (50 per class)
- **Test**: 21,100 images (100 per class)

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
     - H×W×3 RGB image
   * - ``label``
     - int
     - Class label (0-210)

Usage Example
-------------

**Basic Usage**

.. code-block:: python

    from stable_datasets.images.country211 import Country211

    # First run will download + prepare cache, then return the split as a HF Dataset
    ds = Country211(split="train")

    # If you omit the split (split=None), you get a DatasetDict with all available splits
    ds_all = Country211(split=None)

    sample = ds[0]
    print(sample.keys())  # {"image", "label"}

    # Optional: make it PyTorch-friendly
    ds_torch = ds.with_format("torch")

References
----------

- Official website: https://github.com/openai/CLIP/blob/main/data/country211.md
- License: MIT License

Citation
--------

.. code-block:: bibtex

    @inproceedings{radford2021learning, 
    title     = {Learning transferable visual models from natural language supervision},
    author    = {Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
    booktitle = {International conference on machine learning},
    pages     = {8748--8763},
    year      = {2021},
    organization = {PmLR} }



