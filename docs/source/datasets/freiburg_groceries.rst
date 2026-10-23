Freiburg Groceries
==================

Overview
--------

``FreiburgGroceries`` loads the authors' released archive and original train/test
lists for 25 grocery categories. Although the paper describes approximately 5,000
images, the downloadable archive contains **4,947 RGB PNG images**, each 256 x 256.
The file is named ``.tar.gz`` but is an uncompressed tar; the loader detects its
actual format.

.. figure:: teasers/freiburg_groceries_teaser.png
   :align: center
   :width: 90%
   :alt: Five labeled Freiburg Groceries training images from different classes.

   Five training examples from the Freiburg Groceries release by Jund, Abdo,
   Eitel, and Burgard (2016). The preview uses the authors' images and class
   names. See `Source and terms`_ for the original archive, attribution, and
   research-use conditions.

Official partitions
-------------------

Select one of five author configurations, ``split0`` through ``split4``.
``split0`` is the default. Both ``train`` and ``test`` are available; there is no
official validation split. Labels and split lists are pinned to author repository
commit ``8827d2f61c2838756a069562b1c2ff53568ae33b``.

.. list-table::
   :header-rows: 1

   * - Config
     - Training rows
     - Unique training images
     - Test images
   * - ``split0``
     - 7,425
     - 3,929
     - 1,018
   * - ``split1``
     - 7,425
     - 3,989
     - 958
   * - ``split2``
     - 7,425
     - 3,962
     - 977
   * - ``split3``
     - 7,425
     - 3,937
     - 995
   * - ``split4``
     - 7,425
     - 3,943
     - 999

Training lists repeat images; the loader preserves **every row and its original
order**. Train and test image paths are disjoint within each configuration.
Configurations 2--4 omit a small number of archive images; the loader follows the
published lists rather than adding unlisted images. Different configurations are
alternative experiments and must not be concatenated as disjoint datasets.

For an additional validation set, group by ``image_id`` before splitting, so all
copies of an image remain together. The SSL example first deduplicates the
``split0`` training list, then creates a stratified 80/20 train/validation split
using seed 42 (3,143 / 786 unique images). This validation partition is defined by
the example, not by the authors. The 1,018 official test images remain untouched.
The release does not provide physical product instance IDs, so path-disjointness
does not establish independence of real-world product instances.

Fields and labels
-----------------

* ``image``: RGB PIL image by default; float32 CHW tensor with ``with_format("torch")``.
* ``label``: integer 0--24, using the authors' ``classid.txt`` order below.
* ``image_id``: relative source path, for example ``TEA/TEA0123.png``.

The label order is **not alphabetical**:

.. code-block:: text

    0 CANDY       1 JUICE       2 VINEGAR      3 OIL         4 CHOCOLATE
    5 PASTA       6 RICE        7 MILK         8 SPICES      9 HONEY
    10 JAM        11 NUTS       12 CHIPS       13 SODA       14 COFFEE
    15 BEANS      16 TEA        17 CORN        18 CEREAL     19 CAKE
    20 SUGAR      21 WATER      22 FLOUR       23 TOMATO_SAUCE  24 FISH

Usage
-----

.. code-block:: python

    from torch.utils.data import DataLoader
    from stable_datasets.images import FreiburgGroceries

    data = FreiburgGroceries(
        config_name="split0",
        download_dir="./data/downloads",
        processed_cache_dir="./data/processed",
    )
    batch = next(iter(DataLoader(data["train"].with_format("torch"), batch_size=8)))
    assert batch["image"].shape == (8, 3, 256, 256)

The shared builder caches each configuration separately. Source archives are
SHA-256 checked on download. Parsing checks the class vocabulary, path/label
agreement, train/test overlap, duplicate test entries, and missing or non-regular
image members. Archive paths are never extracted onto the filesystem.

Source and terms
----------------

* `Author homepage and terms <http://www2.informatik.uni-freiburg.de/~eitel/freiburg_groceries_dataset.html>`_
* `Author code and split lists <https://github.com/PhilJd/freiburg_groceries_dataset/tree/8827d2f61c2838756a069562b1c2ff53568ae33b>`_
* `Image archive <http://aisdatasets.informatik.uni-freiburg.de/freiburg_groceries_dataset/freiburg_groceries_dataset.tar.gz>`_
* Images SHA-256: ``35c18d03bb6ed057f12a3c5aea0ac5eef9b8e1d5965d1e32142b7d1f1469e3f0``.
* Annotations SHA-256: ``b07ba6e6c3760cb909f4b92de41f7ab505c3a910f8736613ea4b9712537aafb9``.

The authors permit research use, require consent for commercial use, and request
citation of their paper. The source image server was accessible over its published
HTTP URL; HTTPS was unavailable during verification. A pinned checksum verifies
the downloaded bytes. These terms are separate from the library's software license.
The five attributed preview images above are included for dataset documentation;
the full source image archive is not included.

Citation
--------

.. code-block:: bibtex

    @article{jund2016groceries,
      title={The Freiburg Groceries Dataset},
      author={Jund, Philipp and Abdo, Nichola and Eitel, Andreas and Burgard, Wolfram},
      journal={arXiv preprint arXiv:1611.05799},
      year={2016}
    }

Validation
----------

Run ``examples/validate_object_datasets.py`` for the real-data checks, or
``examples/object_datasets_lejepa.py`` for SSL and frozen linear evaluation.
The complete protocol is in ``docs/object-datasets-validation.md``.
