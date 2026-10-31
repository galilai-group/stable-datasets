Scene8
======

The original Spatial Envelope collection contains 2,688 RGB outdoor scene images
in eight classes, each 256 x 256 pixels. The builder preserves original JPEG
bytes, labels, and image IDs. It exposes the complete collection as ``train``;
this is not an official training partition.

.. figure:: teasers/scene8_teaser.png
   :alt: Eight original Scene8 images, one per outdoor scene class, with class names and image IDs.
   :width: 100%

   One example per class, selected by the first image ID in each class.
   Source images are resized for display without cropping or changing labels.

Data Structure
--------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Key
     - Type
     - Description
   * - ``image``
     - PIL.Image.Image
     - RGB PIL image, 256 x 256 pixels
   * - ``label``
     - int
     - Integer class ID (0--7)
   * - ``image_id``
     - str
     - Original filename stem, unique within the dataset

Usage
-----

.. code-block:: python

   from stable_datasets.images import Scene8

   ds = Scene8(split="train")
   sample = ds[0]  # image: PIL image, label: int, image_id: str
   print(ds.features["label"].names)

   # Standard library options control cache locations.
   ds = Scene8(split="train", download_dir="/data/downloads",
               processed_cache_dir="/data/processed")

Class IDs follow the fixed order ``coast``, ``forest``, ``highway``, ``insidecity``,
``mountain``, ``opencountry``, ``street``, and ``tallbuilding``. Rows are sorted
by image ID. Standard caching, formatting, and transforms are supported.

Testing
-------

The tests download the original release through the standard dataset API and
check split sizes, class counts, image decoding, stable IDs, and cache reopening.
Offline parser tests additionally check invalid inputs and label matching.

Source and terms
----------------

Original page and download:
https://people.csail.mit.edu/torralba/code/spatialenvelope/

The original page does not specify a dataset-wide license. The library's code
license does not grant a license to redistribute the images. Download checksums
identify the locally audited release rather than publisher signatures.

Citation
--------

.. code-block:: bibtex

   @article{oliva2001modeling,
       title={Modeling the shape of the scene: A holistic representation of the spatial envelope},
       author={Oliva, Aude and Torralba, Antonio},
       journal={International Journal of Computer Vision},
       volume={42}, pages={145--175}, year={2001}
   }

Known exact-pixel duplicates are retained as source records. Evaluation code
should group duplicate images within partitions and avoid train/test leakage.
