Describable Textures Dataset (DTD)
========

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Image%20Classification-blue" alt="Task: Image Classification">
   <img src="https://img.shields.io/badge/Classes-47-green" alt="Classes: 47">
   <img src="https://img.shields.io/badge/Size-HxWx3-orange" alt="Image Size: HxWx3">
   </p>

Overview
--------
Describable Textures Dataset (DTD) is a texture database, consisting of 5640 images, organized according to a list of 47 terms (categories) inspired from human perception. There are 120 images for each category. Image sizes range between 300x300 and 640x640, and the images contain at least 90% of the surface representing the category attribute. The images were collected from Google and Flickr by entering our proposed attributes and related terms as search queries. The images were annotated using Amazon Mechanical Turk in several iterations. For each image we provide key attribute (main category) and a list of joint attributes.


- **Train**: 1880 images
- **Validation**: 1880 images
- **Test**: 1880 images

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
     - Class label (0-46)

Usage Example
-------------

**Basic Usage**

.. code-block:: python

    from stable_datasets.images.dtd import DTD

    # First run will download + prepare cache, then return the split as a HF Dataset
    ds = DTD(split="train")

    # If you omit the split (split=None), you get a DatasetDict with all available splits
    ds_all = DTD(split=None)

    sample = ds[0]
    print(sample.keys())  # {"image", "label"}

    # Optional: make it PyTorch-friendly
    ds_torch = ds.with_format("torch")

References
----------

- Official website: https://www.robots.ox.ac.uk/~vgg/data/dtd/
- License: MIT License

Citation
--------

.. code-block:: bibtex

    @InProceedings{cimpoi14describing,
        Author    = {M. Cimpoi and S. Maji and I. Kokkinos and S. Mohamed and and A. Vedaldi},
        Title     = {Describing Textures in the Wild},
        Booktitle = {Proceedings of the {IEEE} Conf. on Computer Vision and Pattern Recognition ({CVPR})},
        Year      = {2014}}
