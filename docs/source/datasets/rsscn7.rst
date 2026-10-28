RSSCN7
======

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Image%20Classification-blue" alt="Task: Image Classification">
   <img src="https://img.shields.io/badge/Classes-7-green" alt="Classes: 7">
   <img src="https://img.shields.io/badge/Size-400x400-orange" alt="Image Size: 400x400">
   <img src="https://img.shields.io/badge/Format-RGB-lightgrey" alt="Format: RGB">
   </p>

Overview
--------

The **RSSCN7** (Remote Sensing Scene Classification with 7 classes) dataset was created for the paper "Deep Learning Based Feature Selection for Remote Sensing Scene Classification" by Zou et al. (IEEE Geoscience and Remote Sensing Letters, 2015).

This dataset features **2,800 images** extracted from Google Earth with a resolution of **400×400 pixels**, covering 7 typical scene categories in remote sensing imagery:

- **aGrass**: Grassland scenes
- **bField**: Agricultural field/farmland scenes
- **cIndustry**: Industrial and commercial area scenes
- **dRiverLake**: River and lake/water body scenes
- **eForest**: Forest scenes
- **fResident**: Residential area scenes
- **gParking**: Parking lot scenes

Each class contains **400 images**, making it a balanced dataset ideal for scene classification tasks.

No official split is provided, so default split is set as follows:

- **Train**: 2,800 images (400 per class)

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
     - 400×400 RGB image of a remote sensing scene
   * - ``label``
     - int
     - Class label (0-6) corresponding to the 7 scene categories

Usage Example
-------------

**Basic Usage**

.. code-block:: python

    from stable_datasets.images.rsscn7 import RSSCN7

    # Load the train split
    ds = RSSCN7(split="train")

    # If you omit the split (split=None), you get a DatasetDict with all available splits
    ds_all = RSSCN7(split=None)

    sample = ds[0]
    print(sample.keys())  # {"image", "label"}

    # Access the image and label
    image = sample["image"]  # PIL.Image.Image
    label = sample["label"]  # int (0-6)

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

    ds = RSSCN7(split="train")
    sample = ds[0]
    tensor = transform(sample["image"])
    print(f"Tensor shape: {tensor.shape}")  # torch.Size([3, 400, 400])

Related Datasets
----------------

- :doc:`resisc45`: NWPU-RESISC45 with 45 remote sensing scene classes
- **AID**: Aerial Image Dataset with 30 scene classes
- **UC Merced**: 21-class land use classification dataset
- **PatternNet**: Large-scale remote sensing dataset with 38 classes

References
----------

- Official repository: https://github.com/palewithout/RSSCN7
- Papers with Code: https://paperswithcode.com/dataset/rsscn7
- Also available on Figshare: https://figshare.com/articles/dataset/RSSCN7_Image_dataset/7006946

Citation
--------

.. code-block:: bibtex

    @article{zou2015deep,
        title={Deep Learning Based Feature Selection for Remote Sensing Scene Classification},
        author={Zou, Qin and Ni, Lihao and Zhang, Tong and Wang, Qian},
        journal={IEEE Geoscience and Remote Sensing Letters},
        volume={12},
        number={11},
        pages={2321--2325},
        year={2015},
        publisher={IEEE}
    }
