Sewer-ML
========

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Multi--label%20Defect%20Classification-blue" alt="Task: Multi-label Defect Classification">
   <img src="https://img.shields.io/badge/Images-1.3M-green" alt="Images: 1.3M">
   <img src="https://img.shields.io/badge/Download-~340%20GB-red" alt="Download: ~340 GB">
   <img src="https://img.shields.io/badge/Access-Credentials%20required-important" alt="Access: Credentials required">
   </p>

Overview
--------

Sewer-ML contains 1.3 million images from 75,618 sewer pipe inspection videos, annotated
by professional sewer inspectors across 17 defect classes. Each image carries a multi-hot
vector over those classes plus a binary ``Defect`` flag.

.. note::

   This page carries no teaser figure: ``generate_teaser.py`` has to load the
   dataset, and Sewer-ML is only reachable with the download password. Run
   ``python generate_teaser.py --name SewerML --split validation --num-samples 6
   --output docs/source/datasets/teasers/sewer_ml_teaser.png`` once your
   ``~/.netrc`` entry is in place to add one.

.. warning::

   The archives total roughly **340 GB** (14 train + 2 validation + 2 test, ~19 GB each),
   and the host ignores HTTP ``Range``, so an interrupted archive restarts from zero.
   Point ``download_dir=`` and ``processed_cache_dir=`` at a volume with room before
   loading. Splits are fetched one at a time, so a finished split's cache is committed
   before the next archive starts.

Access and the download password
--------------------------------

Sewer-ML is hosted on `sciencedata.dk <https://sciencedata.dk>`_ as a password-protected
share. Request the download password with the Google form linked from
https://vap.aau.dk/sewer-ml/ (https://forms.gle/hBaPtoweZumZAi4u9).

**There is no username** — the share validates the password only. So the netrc entry
needs just a ``password`` line. stable-datasets requires no further configuration:
``requests`` reads ``~/.netrc`` natively, so the existing download path authenticates
without any extra arguments.

.. code-block:: text

    machine sciencedata.dk
    password <the-download-password>

Then ``chmod 600 ~/.netrc``. If you already have a ``~/.netrc``, append this block rather
than overwriting the file. A ``login`` line is accepted but ignored, so any placeholder
value works if a tool in your stack insists on one.

A refused download raises ``PermissionError`` carrying these instructions rather than a
generic download failure.

To keep the password outside your home directory — on a shared machine, or in CI — point
the ``NETRC`` environment variable at the file instead:

.. code-block:: bash

    NETRC=/secure/path/sewerml_netrc python train.py

Alternatively, point the builder at an already-downloaded copy:

.. code-block:: python

    ds = SewerML(split="train", data_dir="/path/to/sewer-ml")

Both a flat directory and the ``zips/`` layout produced by the upstream download script
are accepted.

Splits
------

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - Split
     - Annotations
     - Archives
     - Labeled
   * - ``train``
     - ``SewerML_Train.csv``
     - ``train00``–``train13``
     - yes
   * - ``validation``
     - ``SewerML_Val.csv``
     - ``valid00``–``valid01``
     - yes
   * - ``test``
     - ``SewerML_Test.csv``
     - ``test00``–``test01``
     - **no**

.. note::

   The benchmark **withholds the test annotations** — ``SewerML_Test.csv`` lists
   filenames only. Rows in the test split therefore have ``label``, ``defects`` and
   ``water_level`` set to ``None``. Use the validation split for labeled evaluation.

Data Structure
--------------

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Key
     - Type
     - Description
   * - ``image``
     - ``PIL.Image.Image``
     - The inspection frame
   * - ``label``
     - int or ``None``
     - Binary ``Defect`` flag; ``None`` on the test split
   * - ``defects``
     - list[int] or ``None``
     - Multi-hot vector over the 17 defect classes; ``None`` on the test split
   * - ``water_level``
     - int or ``None``
     - Water level annotation; ``None`` on the test split
   * - ``filename``
     - str
     - Image filename as listed in the annotation CSV

Defect classes, in vector order: ``RB``, ``OB``, ``PF``, ``DE``, ``FS``, ``IS``, ``RO``,
``IN``, ``AF``, ``BE``, ``FO``, ``GR``, ``PH``, ``PB``, ``OS``, ``OP``, ``OK``.

Usage Example
-------------

.. code-block:: python

    from stable_datasets.images import SewerML
    from stable_datasets.images.sewer_ml import SEWER_ML_DEFECT_CLASSES

    val = SewerML(split="validation")

    sample = val[0]
    print(sample["label"])     # 0 or 1
    print(sample["defects"])   # length-17 multi-hot vector

    # Which defect codes are present on this image
    present = [c for c, v in zip(SEWER_ML_DEFECT_CLASSES, sample["defects"]) if v]
    print(present)

.. note::

   Images are read directly out of the zip archives, so no extracted copy is written to
   disk — this saves roughly 340 GB compared with unzipping first.

References
----------

- Official website: https://vap.aau.dk/sewer-ml/
- License: CC BY-NC-SA 4.0 (non-commercial research use)

Citation
--------

.. code-block:: bibtex

    @inproceedings{haurum2021sewer,
      title={Sewer-ML: A Multi-Label Sewer Defect Classification Dataset and Benchmark},
      author={Haurum, Joakim Bruslund and Moeslund, Thomas B},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      pages={13456--13467},
      year={2021}
    }
