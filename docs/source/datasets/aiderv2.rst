AIDERv2
=======

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Disaster%20Image%20Classification-blue" alt="Task: Disaster Image Classification">
   <img src="https://img.shields.io/badge/Classes-4-green" alt="Classes: 4">
   <img src="https://img.shields.io/badge/Size-Variable-orange" alt="Image Size: Variable">
   </p>

Overview
--------

AIDERv2 is an aerial image classification dataset for recognizing natural disasters and non-disaster scenes. It contains 16,723 images across four classes: earthquake, flood, normal, and wildfire. The dataset extends the original AIDER dataset with additional open-source imagery, including frames extracted from YouTube videos.

- **Train**: 13,399 images
- **Validation**: 1,670 images
- **Test**: 1,654 images

.. image:: teasers/aiderv2_teaser.png
   :align: center
   :width: 90%

Data Structure
--------------

When accessing an example using ``ds[i]``, you will receive a dictionary with the following keys:

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Key
     - Type
     - Description
   * - ``image``
     - ``PIL.Image.Image``
     - Aerial image at its original dimensions
   * - ``label``
     - int
     - Class label (0-3), corresponding to the class mapping below

Class Labels
^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Label
     - Class
     - Description
   * - 0
     - Earthquake
     - Images depicting earthquake-related damage
   * - 1
     - Flood
     - Images depicting flooding
   * - 2
     - Normal
     - Images without a depicted disaster
   * - 3
     - Wildfire
     - Images depicting wildfire or fire

Usage Example
-------------

**Basic Usage**

.. code-block:: python

   from stable_datasets.images.aiderv2 import AIDERv2

   # Load individual splits
   train_ds = AIDERv2(split="train")
   val_ds = AIDERv2(split="validation")
   test_ds = AIDERv2(split="test")

   sample = train_ds[0]
   print(sample.keys())  # {"image", "label"}

   # Inspect class names
   print(train_ds.info.features["label"].names)

   # Optional: make it PyTorch-friendly
   train_torch = train_ds.with_format("torch")

References
----------

- Dataset: https://zenodo.org/records/10891054
- License: Creative Commons Attribution 4.0 International

Citation
----------

.. code-block:: bibtex

   @dataset{shianios_kyrkou_2024_aiderv2,
     author = {Shianios, D. and Kyrkou, C.},
     title = {AIDERv2 (Aerial Image Dataset for Emergency Response Applications)},
     year = {2024},
     publisher = {Zenodo},
     doi = {10.5281/zenodo.10891054},
     url = {https://doi.org/10.5281/zenodo.10891054}
   }
 
    @inproceedings{shianios2023benchmark,
      title={A Benchmark and Investigation of Deep-Learning-Based Techniques for Detecting Natural Disasters in Aerial Images},
      author={Shianios, D. and Kyrkou, C. and Kolios, P. S.},
      booktitle={Computer Analysis of Images and Patterns (CAIP)},
      series={Lecture Notes in Computer Science},
      volume={14185},
      publisher={Springer},
      year={2023},
      doi={10.1007/978-3-031-44240-7_24}
    }
 
    @article{shianios2024direcnetv2,
      title={DiRecNetV2: A Transformer-Enhanced Network for Aerial Disaster Recognition},
      author={Shianios, D. and Kolios, P. and Kyrkou, C.},
      journal={SN Computer Science},
      year={2024}
    }