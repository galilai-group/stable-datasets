CC3M (Conceptual Captions)
=============================

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Image%20Captioning-blue" alt="Task: Image Captioning">
   <img src="https://img.shields.io/badge/Image%20Size-Variable-orange" alt="Image Size: Variable">
   </p>

Overview
--------

The CC3M dataset is a large dataset of images with corresponding descriptions scraped from across the Internet using a purely automatic pipeline. The pipeline was developed by Google AI to filter out many of the data items gathered and keep the data quality high, so the original dataset contained around 3.3M images for training and around 15.8k images for cross-validation. The content of the images and the style of the descriptions vary heavily. These data are typically used for tasks like pretraining or fine-tuning vision-language models, such as for automatic image captioning.

**Note:** Many images through the dataset have become unavailable due to their original owners taking them down. As of Jan. 25, 2026, the dataset sizes are:

- **Train**: ~1.6M images
- **Validation**: ~8.6k images

.. image:: teasers/cc3m_teaser.png
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
     - Variable-resolution RGB image
   * - ``caption``
     - ``str``
     - Free-form text describing the image

Usage Example
-------------

**Basic Usage**

.. code-block:: python

    from stable_datasets.images.cc3m import CC3M

    # First run will download + prepare cache, then return the split as a HF Dataset
    ds_train = CC3M(split="train")
    ds_valid = CC3M(split="validation")

    # If you omit the split (split=None), you get a DatasetDict with all available splits
    ds_all = CC3M(split=None)

    # You can easily examine what information is available in each sample
    sample = ds_train[0]
    print(sample.keys())  # {"image", "caption"}
    print(f"Caption: {sample['caption']}")  # e.g., "dry fields on a summer day"

    # Optional: Make it PyTorch-friendly
    ds_train_torch = ds_train.with_format("torch")

References
----------

- Official website: https://ai.google.com/research/ConceptualCaptions/
- Download page: https://ai.google.com/research/ConceptualCaptions/download

Citation
--------

.. code-block:: bibtex

    @inproceedings{sharma-etal-2018-conceptual,
        title = "Conceptual Captions: A Cleaned, Hypernymed, Image Alt-text Dataset For Automatic Image Captioning",
        author = "Sharma, Piyush  and Ding, Nan  and Goodman, Sebastian  and Soricut, Radu",
        editor = "Gurevych, Iryna  and Miyao, Yusuke",
        booktitle = "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
        month = jul,
        year = "2018",
        address = "Melbourne, Australia",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/P18-1238/",
        doi = "10.18653/v1/P18-1238",
        pages = "2556--2565",
    }
