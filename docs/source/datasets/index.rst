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
   cifar10
   cifar100
   cifar10_c
   cifar100_c
   cars196
   dtd
   medmnist

.. note::
   Documentation is being added progressively, as datasets are ready for usage. Please only use datasets found in the documentation.
