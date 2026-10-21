Rendered SST2
=============

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Image%20Classification-blue" alt="Task: Image Classification">
   <img src="https://img.shields.io/badge/Classes-2-green" alt="Classes: 2">
   <img src="https://img.shields.io/badge/Size-448x448-orange" alt="Image Size: 448x448">
   </p>

Overview
--------

Rendered SST2 is an image classification benchmark introduced with CLIP to evaluate optical character recognition. Each image is a sentence from the Stanford Sentiment Treebank v2 (SST-2) rendered as black text on a white 448×448 RGB canvas, labelled with its binary sentiment: *negative* or *positive*. Solving the task requires reading the text, so models without OCR ability score close to chance (50%).

- **Train**: 6,920 images (3,310 negative / 3,610 positive)
- **Validation**: 872 images (428 negative / 444 positive)
- **Test**: 1,821 images (912 negative / 909 positive)

The official partitions are kept as released. Because SST-2 contains a few repeated sentences, 9 training images are pixel-identical to another training image and 2 test images also appear in the training split.

.. image:: teasers/rendered_sst2_teaser.png
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
     - 448×448 RGB image of a rendered sentence
   * - ``label``
     - int
     - Sentiment label (0: negative, 1: positive)

Usage Example
-------------

**Basic Usage**

.. code-block:: python

    from stable_datasets.images.rendered_sst2 import RenderedSST2

    # First run will download + prepare cache, then return the split as a HF Dataset
    ds_train = RenderedSST2(split="train")
    ds_val = RenderedSST2(split="valid")
    ds_test = RenderedSST2(split="test")

    # If you omit the split (split=None), you get a DatasetDict with all available splits
    ds_all = RenderedSST2(split=None)

    sample = ds_train[0]
    print(sample.keys())  # {"image", "label"}
    print(f"Label: {sample['label']}")  # 0 (negative) or 1 (positive)

    # Optional: make it PyTorch-friendly
    ds_train_torch = ds_train.with_format("torch")

References
----------

- Dataset card: https://github.com/openai/CLIP/blob/main/data/rendered-sst2.md
- Stanford Sentiment Treebank: https://nlp.stanford.edu/sentiment/

Citation
--------

.. code-block:: bibtex

    @inproceedings{radford2021learning,
      title={Learning transferable visual models from natural language supervision},
      author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
      booktitle={International conference on machine learning},
      pages={8748--8763},
      year={2021},
      organization={PMLR}
    }

    @inproceedings{socher2013recursive,
      title={Recursive deep models for semantic compositionality over a sentiment treebank},
      author={Socher, Richard and Perelygin, Alex and Wu, Jean and Chuang, Jason and Manning, Christopher D and Ng, Andrew Y and Potts, Christopher},
      booktitle={Proceedings of the 2013 conference on empirical methods in natural language processing},
      pages={1631--1642},
      year={2013}
    }
