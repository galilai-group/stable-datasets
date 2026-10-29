Datasets
========

This section provides detailed documentation for all datasets available in stable-datasets.

Overview
--------

stable-datasets provides easy access to a wide variety of datasets for machine learning research, with a focus on stability and reproducibility. Each dataset page includes:

- **Example Samples**: Visual examples or data snippets from the dataset
- **Dataset Details**: Number of classes, target types, and data specifications
- **Data Structure**: Keys and data types returned when accessing the dataset
- **Usage Examples**: Code snippets showing how to load and use the dataset
- **Related Datasets**: Links to similar or derived datasets
- **Citation**: The original paper to cite when using the dataset

Getting Started
---------------

All datasets can be loaded using the same consistent API:

.. code-block:: python

    from stable_datasets.images.<dataset_module> import <DatasetClass>

    # First run will download + prepare cache, then return the split as a HF Dataset
    ds = <DatasetClass>(split="train")

    # If you omit the split (split=None), you get a DatasetDict with all available splits
    ds_all = <DatasetClass>(split=None)

    # Access individual examples
    sample = ds[0]
    print(sample.keys())  # e.g., {"image", "label"}

    # Optional: make it PyTorch-friendly
    ds_torch = ds.with_format("torch")

Available Datasets
------------------

.. toctree::
   :maxdepth: 1
   :caption: Image Classification Datasets

   arabic_characters
   arabic_digits
   caltech256
   cifar10
   cifar100
   cifar10_c
   cifar100_c
   cinic10
   cars196
   cct20
   coil100
   dtd
   not_mnist
   fashion_mnist
   k_mnist
   k_mnist49
   medmnist
   patch_camelyon
   awa2
   beans
   deep_weeds
   stl10
   tiny_imagenet
   tiny_imagenet_c
   imagenet_v2
   imagenet_sketch
   imagenet_a
   imagenet_r
   e_mnist
   fgvc_aircraft
   flowers17
   flowers102
   tf_flowers
   imagewoof
   food101
   fmd
   cub200
   stanford_dogs
   country211
   galaxy10
   hasy_v2
   face_pointing
   rock_paper_scissor
   svhn
   linnaeus5
   malaria
   kather_colorectal_histology
   sun397
   stanford40
   usps
   oxford_iiit_pet
   eurosat

.. toctree::
   :maxdepth: 1
   :caption: Time Series Forecasting Datasets

   electricity_hourly
   uber_tlc

.. toctree::
   :maxdepth: 1
   :caption: Video Datasets

   something_something_v2
   clevrer

.. toctree::
   :maxdepth: 1
   :caption: Disentanglement Datasets

   dsprites
   dsprites_color
   dsprites_noise
   dsprites_scream
   shapes3d
   cars3d
   small_norb

.. note::
   Documentation is being added progressively, as datasets are ready for usage. Please only use datasets found in the documentation.
