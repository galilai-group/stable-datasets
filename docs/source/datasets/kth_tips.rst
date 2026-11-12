KTH-TIPS
========

Overview
--------

.. image:: teasers/kth_tips_teaser.png
   :align: center
   :width: 90%

KTH-TIPS contains 810 color images of 10 materials photographed under
different scales, poses, and lighting conditions. Each material has 81
images. The cropped color images are 200 × 200 pixels, except 63 that are
not square (154-277 × 113-193 pixels): brown bread at scales 6-9 and cracker
at scales 7-9, which were too small to crop at 200 × 200.

The source dataset does not provide an official train/test split. The
``train`` split in this builder contains all 810 images.

Data Structure
--------------

Each ``ds[i]`` returns:

.. list-table::
   :header-rows: 1

   * - Key
     - Type
     - Description
   * - ``image``
     - ``PIL.Image.Image``
     - RGB image, 200 × 200 except for 63 images (see Overview)
   * - ``label``
     - ``int``
     - Material class ID from 0 to 9
   * - ``scale``
     - ``int``
     - Scale identifier from 1 to 9
   * - ``image_id``
     - ``int``
     - Image identifier from 1 to 9 within that scale

Usage Example
-------------

.. code-block:: python

    from stable_datasets.images import KTHTIPS

    ds = KTHTIPS(split="train")
    print(len(ds))  # 810

    sample = ds[0]
    print(sample["image"].size, sample["label"], sample["scale"])

License and Terms of Use
------------------------

No explicit dataset license or reuse terms were identified on the
`official KTH-TIPS homepage
<https://www.csc.kth.se/cvap/databases/kth-tips/>`_,
`download page
<https://www.csc.kth.se/cvap/databases/kth-tips/download.html>`_, or in the
`KTH-TIPS user guide
<https://www.csc.kth.se/cvap/databases/kth-tips/kth_tips.pdf>`_
(checked October 29, 2026).

The user guide describes the cropped database as freely available online,
but does not specify a license granting reuse rights. For clarification
about permitted uses, contact Eric Hayman at hayman@nada.kth.se, as listed
on the official download page.

The stable-datasets repository license does not establish a license for
the downloaded dataset.

Source and Citation
-------------------

- Source: https://www.csc.kth.se/cvap/databases/kth-tips/
- M. Fritz, E. Hayman, B. Caputo, and J.-O. Eklundh,
  *The KTH-TIPS Database*, 2004.
