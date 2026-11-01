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
   cifar10_1
   cifar10_2
   cifar100
   cifar10_c
   cifar100_c
   cinic10
   citrus_leaves
   cars196
   coil20
   cct20
   coil100
   dtd
   not_mnist
   fashion_mnist
   fer2013
   kitti_distance
   oracle_mnist
   rendered_sst2
   k_mnist
   kannada_mnist
   devanagari
   kth_tips
   k_mnist49
   medmnist
   patch_camelyon
   awa2
   beans
   deep_weeds
   stl10
   tiny_imagenet
   tiny_imagenet_c
   trashnet
   imagenet_v2
   imagenet_sketch
   imagenet_a
   imagenet_r
   e_mnist
   eth80
   mnist_c
   fgvc_aircraft
   flowers17
   flowers102
   tf_flowers
   imagewoof
   food101
   freiburg_groceries
   fmd
   cub200
   stanford_dogs
   country211
   cmu_faces
   defungi
   galaxy10
   gtsrb
   hasy_v2
   face_pointing
   olivetti_faces
   optdigits
   rock_paper_scissor
   svhn
   linnaeus5
   caltech101
   mnist_m
   bee_dataset
   aiderv2
   grocery_store
   plantdoc
   fruits360
   belgium_tsc
   plant_village
   intel_image_classification
   horse_or_human
   semeion
   malaria
   kather_colorectal_histology
   sun397
   stanford40
   usps
   oxford_iiit_pet
   eurosat
   breakhis
   patternnet
   pacs
   office_home
   caltech101_silhouettes
   scene8
   isic2016
   brain_tumor_mri
   lc25000

.. toctree::
   :maxdepth: 1
   :caption: Image Segmentation Datasets

   weizmann_seg
   bsds500

.. toctree::
   :maxdepth: 1
   :caption: Time Series Classification Datasets

   basic_motions
   epilepsy
   hand_movement_direction
   natops
   racket_sports
   uwave_gesture_library

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

.. toctree::
   :maxdepth: 1
   :caption: Wearable Time-Series Datasets

   uci_har
   mhealth

.. note::
   Documentation is being added progressively, as datasets are ready for usage. Please only use datasets found in the documentation.

.. toctree::
   :maxdepth: 1
   :caption: Sensor Time Series

   robot_execution_failures
   daily_and_sports_activities
