RESISC45
========

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Image%20Classification-blue" alt="Task: Image Classification">
   <img src="https://img.shields.io/badge/Classes-45-green" alt="Classes: 45">
   <img src="https://img.shields.io/badge/Images-31%2C500-yellow" alt="Images: 31,500">
   <img src="https://img.shields.io/badge/Size-256x256-orange" alt="Image Size: 256x256">
   <img src="https://img.shields.io/badge/Format-RGB-lightgrey" alt="Format: RGB">
   </p>

Overview
--------

**RESISC45** (also known as **NWPU-RESISC45**) is a publicly available benchmark dataset for Remote Sensing Image Scene Classification (RESISC), created by Northwestern Polytechnical University (NWPU). The dataset was introduced in the paper "Remote Sensing Image Scene Classification: Benchmark and State of the Art" by Cheng et al. (Proceedings of the IEEE, 2017).

This large-scale dataset contains **31,500 images** covering **45 scene classes** with **700 images per class**. All images are:

- **256×256 pixels** in resolution
- **RGB** (three spectral bands)
- Manually extracted from **Google Earth** imagery
- Sourced from **over 100 countries and regions**
- Pixel resolution ranges from **0.2 to 30 meters per pixel**
- High variability in **resolution, weather conditions, and illumination**

The 45 scene classes include diverse remote sensing scenarios such as airports, beaches, bridges, forests, harbors, residential areas, stadiums, and more.

No official split is provided, so default split is set as follows:

- **Train**: 31,500 images 

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
     - 256×256 RGB image of a remote sensing scene
   * - ``label``
     - int
     - Class label (0-44) corresponding to one of 45 scene categories

Class Labels
------------

The 45 scene classes are:

.. code-block:: python

    [
        "airplane", "airport", "baseball_diamond", "basketball_court", "beach",
        "bridge", "chaparral", "church", "circular_farmland", "cloud",
        "commercial_area", "dense_residential", "desert", "forest", "freeway",
        "golf_course", "ground_track_field", "harbor", "industrial_area", "intersection",
        "island", "lake", "meadow", "medium_residential", "mobile_home_park",
        "mountain", "overpass", "palace", "parking_lot", "railway",
        "railway_station", "rectangular_farmland", "river", "roundabout", "runway",
        "sea_ice", "ship", "snowberg", "sparse_residential", "stadium",
        "storage_tank", "tennis_court", "terrace", "thermal_power_station", "wetland"
    ]

Usage Example
-------------

**Basic Usage**

.. code-block:: python

    from stable_datasets.images.resisc45 import RESISC45

    # Load the train split
    ds = RESISC45(split="train")

    # If you omit the split (split=None), you get a DatasetDict with all available splits
    ds_all = RESISC45(split=None)

    sample = ds[0]
    print(sample.keys())  # {"image", "label"}

    # Access the image and label
    image = sample["image"]  # PIL.Image.Image
    label = sample["label"]  # int (0-44)

    # Get the class names
    class_names = ds.features["label"].names
    print(f"Label: {label} -> Class: {class_names[label]}")

**With Transforms**

.. code-block:: python

    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ds = RESISC45(split="train")
    sample = ds[0]
    tensor = transform(sample["image"])
    print(f"Tensor shape: {tensor.shape}")  # torch.Size([3, 256, 256])

**Training a Model**

.. code-block:: python

    from torch.utils.data import DataLoader

    ds_train = RESISC45(split="train")
    ds_val = RESISC45(split="validation")

    # Convert to PyTorch format
    ds_train.set_format("torch")
    ds_val.set_format("torch")

    train_loader = DataLoader(ds_train, batch_size=32, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=32, shuffle=False)

    # Your training loop here
    for batch in train_loader:
        images = batch["image"]
        labels = batch["label"]
        # ...

Requirements
------------

No additional packages are required beyond the standard dependencies.

.. note::

    **Large Download Size**: The RESISC45 dataset is approximately 500MB compressed (ZIP format). The download and extraction may take several minutes depending on your internet connection. The dataset is automatically downloaded from Figshare.

Related Datasets
----------------

- :doc:`rsscn7`: Remote sensing scene classification with 7 classes
- **AID**: Aerial Image Dataset with 30 scene classes and 10,000 images
- **UC Merced**: 21-class land use classification dataset with 2,100 images
- **PatternNet**: 38-class remote sensing dataset with 30,400 images

References
----------

- Official homepage: http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html
- Figshare repository: https://figshare.com/articles/dataset/NWPU-RESISC45/19166525
- TensorFlow Datasets: https://www.tensorflow.org/datasets/catalog/resisc45
- Papers with Code: https://paperswithcode.com/dataset/resisc45

Citation
--------

.. code-block:: bibtex

    @article{cheng2017remote,
        title={Remote sensing image scene classification: Benchmark and state of the art},
        author={Cheng, Gong and Han, Junwei and Lu, Xiaoqiang},
        journal={Proceedings of the IEEE},
        volume={105},
        number={10},
        pages={1865--1883},
        year={2017},
        publisher={IEEE}
    }
