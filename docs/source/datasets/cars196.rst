CARS-196
========

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Image%20Classification-blue" alt="Task: Image Classification">
   <img src="https://img.shields.io/badge/Classes-196-green" alt="Classes: 196">
   <img src="https://img.shields.io/badge/Size-HxWx3-orange" alt="Image Size: HxWx3">
   </p>

Overview
--------
The Cars dataset contains 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split. Classes are typically at the level of Make, Model, Year, e.g. 2012 Tesla Model S or 2012 BMW M3 coupe. Image resolutions vary across samples. No resizing is applied by default.


- **Train**: 8144 images
- **Test**: 8041 images

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
     - Class label (0-195)

Usage Example
-------------

**Basic Usage**

.. code-block:: python

    from stable_datasets.images.cars196 import Cars196

    # First run will download + prepare cache, then return the split as a HF Dataset
    ds = Cars196(split="train")

    # If you omit the split (split=None), you get a DatasetDict with all available splits
    ds_all = Cars196(split=None)

    sample = ds[0]
    print(sample.keys())  # {"image", "label"}

    # Optional: make it PyTorch-friendly
    ds_torch = ds.with_format("torch")

References
----------

- Official website: https://ai.stanford.edu/~jkrause/cars/car_dataset.html
- License: MIT License

Citation
--------

.. code-block:: bibtex

    @inproceedings{krause20133d,
    title={3d object representations for fine-grained categorization},
    author={Krause, Jonathan and Stark, Michael and Deng, Jia and Fei-Fei, Li},
    booktitle={Proceedings of the IEEE international conference on computer vision workshops},
    pages={554--561},
    year={2013}}
