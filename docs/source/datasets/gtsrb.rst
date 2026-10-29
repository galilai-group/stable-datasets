GTSRB
=====

Overview
--------

The German Traffic Sign Recognition Benchmark (GTSRB) is an image classification
dataset with 43 traffic sign classes. The official final distribution has 39,209
training images and 12,630 test images, for 51,839 images in total. There is no
validation split. Labels retain the benchmark's integer class IDs, 0 through 42;
the ``ClassLabel`` names are the corresponding numeric strings.

The original images are PPM files with variable width and height. They are
decoded as RGB without resizing, cropping, padding, or normalization. The
dataset's cache stores them as PNG while preserving their pixels and dimensions.

The training ZIP includes per-class annotation CSVs, but labels are read from
the image class directories. The test image ZIP includes annotations without
labels; the builder matches test images to the separate ground-truth CSV by
filename. Width, height, and sign-region coordinates remain in the source files
and are not exposed as dataset columns.

.. image:: teasers/gtsrb_teaser.png
   :align: center
   :width: 90%

Data Structure
--------------

Each example has these fields:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Key
     - Type
     - Description
   * - ``image``
     - ``PIL.Image.Image``
     - Original-size RGB traffic sign image
   * - ``label``
     - int
     - Official class ID, 0 through 42

Usage Example
-------------

.. code-block:: python

   from stable_datasets.images import GTSRB

   train = GTSRB(split="train")
   test = GTSRB(split="test")

   print(len(train))  # 39209
   print(len(test))   # 12630

   sample = train[0]
   print(sample.keys())
   print(sample["image"].size)
   print(sample["label"])

Source and License
------------------

The `maintained benchmark distribution
<https://github.com/houbensebastian/GermanTrafficSignBenchmarks>`_ links to
the `official ERDA/SID GTSRB archive
<https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html>`_.
The archive provides ``GTSRB_Final_Training_Images.zip``,
``GTSRB_Final_Test_Images.zip``, and ``GTSRB_Final_Test_GT.zip``. The benchmark
downloads moved to external hosting in 2019. The builder tries the original
ERDA URLs first, then raw Wayback Machine copies of the same ZIPs if needed.
Downloaded files are checked against their SHA-256 hashes. The benchmark and
archive do not state a license suitable for the dataset license field, so it
is left empty.

Citation
--------

.. code-block:: bibtex

   @inproceedings{Stallkamp-IJCNN-2011,
     author = {Johannes Stallkamp and Marc Schlipsing and Jan Salmen and Christian Igel},
     title = {The {G}erman {T}raffic {S}ign {R}ecognition {B}enchmark: A multi-class classification competition},
     booktitle = {IEEE International Joint Conference on Neural Networks},
     pages = {1453--1460},
     year = {2011}
   }
