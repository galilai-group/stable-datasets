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
   awa2
   beans
   cars196
   cars3d
   cifar10
   cifar100
   cifar100_c
   cifar10_c
   clevrer
   country211
   cub200
   dsprites
   dsprites_color
   dsprites_noise
   dsprites_scream
   dtd
   e_mnist
   face_pointing
   fashion_mnist
   fgvc_aircraft
   flowers102
   food101
   galaxy10
   hasy_v2
   k_mnist
   linnaeus5
   medmnist
   not_mnist
   rock_paper_scissor
   shapes3d
   small_norb
   stl10
   svhn
   tiny_imagenet
   tiny_imagenet_c

.. toctree::
   :maxdepth: 1
   :caption: Video Datasets

   something_something_v2

.. note::
   Documentation is being added progressively, as datasets are ready for usage. Please only use datasets found in the documentation.
