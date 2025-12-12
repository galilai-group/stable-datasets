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

    from stable_datasets import load_dataset

    # Load a dataset
    dataset = load_dataset('dataset_name')
    
    # Access splits
    train_data = dataset['train']
    test_data = dataset['test']
    
    # Access individual examples
    example = train_data[0]

Available Datasets
------------------

.. toctree::
   :maxdepth: 1
   :caption: Image Classification Datasets

   cifar10

.. note::
   Documentation is being added progressively, as datasets are ready for usage. Please only use datasets found in the documentation.
