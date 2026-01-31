ESC-50
==========

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Environmental%20Sound%20Classification-blue" alt="Task: Environmental Sound Classification">
   <img src="https://img.shields.io/badge/Classes-50-green" alt="Classes: 50">
   <img src="https://img.shields.io/badge/Audio Length-5 seconds-orange" alt="Audio Length: 5 seconds">
   </p>

Overview
--------

The ESC-50 dataset is a dataset of audio clips widely used for sound classification benchmarking. It consists of 2,000 audio clips each of length 5 seconds, and they are distributed across 50 categories (with 40 clips per category). There is only a single dataset with no separate splits.

.. image:: teasers/esc50_teaser.png
   :align: center
   :width: 90%

A sample audio clip from the dataset can be heard at ``docs/source/datasets/teasers/esc50_teaser.wav``.

Data Structure
--------------

When accessing an example using ``ds[i]``, you will receive a dictionary with the following keys:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Key
     - Type
     - Description
   * - ``audio``
     - ``torchcodec.decoders.AudioDecoder``
     - 5 second audio clip loaded from a .wav file
   * - ``fold``
     - ``int``
     - Cross-validation fold index (1-5)
   * - ``category``
     - ``int``
     - Class label (0-49)
   * - ``major_category``
     - ``int``
     - Broader class label (0-4)
   * - ``esc10``
     - ``bool``
     - Whether or not this clip was in the original ESC-10 dataset
   * - ``clip_id``
     - ``int``
     - ID of original audio clip from Freesound
   * - ``take``
     - ``str``
     - Single character denoting which section of the Freesound clip it refers to

Usage Example
-------------

.. code-block:: python

    from stable_datasets.timeseries.esc50 import ESC50

    # First run will download + prepare cache, then return the split as a HF Dataset
    ds = ESC50(split="test")

    sample = ds[0]
    print(sample.keys())  # {"audio", "fold", "category", "major_category", "esc10", "clip_id", "take"}
    print(f"Category: {sample['category']}") # e.g., 36 ("vacuum_cleaner")
    print(f"Major category: {sample['major_category']}") # e.g., 3 ("interior_or_domestic_sounds")

    # Optional: make it PyTorch-friendly
    ds_torch = ds.with_format("torch")

References
----------

- Homepage: https://github.com/karolpiczak/ESC-50

Citation
--------

.. code-block:: bibtex

    @inproceedings{piczak2015dataset,
        title = {{ESC}: {Dataset} for {Environmental Sound Classification}},
        author = {Piczak, Karol J.},
        booktitle = {Proceedings of the 23rd {Annual ACM Conference} on {Multimedia}},
        date = {2015-10-13},
        url = {http://dl.acm.org/citation.cfm?doid=2733373.2806390},
        doi = {10.1145/2733373.2806390},
        location = {{Brisbane, Australia}},
        isbn = {978-1-4503-3459-4},
        publisher = {{ACM Press}},
        pages = {1015--1018}
    }
