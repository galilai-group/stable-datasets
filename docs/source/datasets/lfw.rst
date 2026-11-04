LFW (Labeled Faces in the Wild)
===============================

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Face%20Verification-blue" alt="Task: Face Verification">
   <img src="https://img.shields.io/badge/Classes-2-green" alt="Classes: 2 (same / different)">
   <img src="https://img.shields.io/badge/Size-250x250-orange" alt="Image Size: 250x250">
   </p>

.. image:: teasers/lfw_teaser.png
     :align: center
     :width: 90%

Overview
--------

Labeled Faces in the Wild (LFW) is a database of more than 13,000 face photographs of 5,749 people, collected from the
web and designed for studying unconstrained face recognition. The only constraint on the faces is that they were
detected by the Viola-Jones face detector.

This builder exposes the official face **verification** protocol. Each example is a pair of face images and a label
saying whether both photographs show the same person. It follows LFW's development ("View 1") split:

- **Train**: 2,200 pairs from ``pairsDevTrain.txt`` (1,100 same-person, 1,100 different-person)
- **Test**: 1,000 pairs from ``pairsDevTest.txt`` (500 same-person, 500 different-person)

There is no validation split in the official protocol. The 10-fold ``pairs.txt`` file used for final reported results
is not included. Images are the original (unaligned) 250×250 release. The files are downloaded from the figshare
mirrors that scikit-learn also uses, because the original download is not reliably available.

Data Structure
--------------

When accessing an example using ``ds[i]`` you will receive a dictionary with these keys:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Key
     - Type
     - Description
   * - ``image1``
     - ``PIL.Image.Image``
     - 250×250×3 RGB face image
   * - ``image2``
     - ``PIL.Image.Image``
     - 250×250×3 RGB face image
   * - ``label``
     - int
     - 0 = different people, 1 = same person

Usage Example
-------------

**Basic Usage**

.. code-block:: python

   from stable_datasets.images import LFW

   # First run will download + prepare cache, then return the split
   ds = LFW(split="train")

   # If you omit the split (split=None), you get a DatasetDict with all available splits
   ds_all = LFW(split=None)

   sample = ds[0]
   print(sample.keys())  # {"image1", "image2", "label"}

   # Optional: make it PyTorch-friendly
   ds_torch = ds.with_format("torch")

Related Datasets
----------------

- Other face datasets in the library (see ``celeb_a``, ``face_pointing``)

References
----------

- LFW home page: http://vis-www.cs.umass.edu/lfw/
- Paper: G. B. Huang, M. Ramesh, T. Berg, E. Learned-Miller, "Labeled Faces in the Wild: A Database for Studying Face
  Recognition in Unconstrained Environments", UMass Amherst Technical Report 07-49, 2007.

Citation
--------

.. code-block:: bibtex

   @TechReport{LFWTech,
     author      = {Gary B. Huang and Manu Ramesh and Tamara Berg and Erik Learned-Miller},
     title       = {Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments},
     institution = {University of Massachusetts, Amherst},
     year        = {2007},
     number      = {07-49},
     month       = {October}
   }