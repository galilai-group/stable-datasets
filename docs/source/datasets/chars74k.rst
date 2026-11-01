
Chars74K EnglishHnd
===================

Overview
--------

Chars74K EnglishHnd is a handwritten character recognition
dataset containing 3,410 images across 62 character classes.
The classes include digits (0-9), uppercase letters (A-Z),
and lowercase letters (a-z).

This implementation uses the first official train/test split
provided with the Chars74K dataset.

Dataset statistics:

- Total images: 3,410
- Training samples (official split 1): 310
- Test samples (official split 1): 1,674
- Number of classes: 62
- Original images: variable resolution
- Task: image classification

The official split does not use all 3,410 images.
The remaining images are not assigned to the selected
training or test split.

.. image:: teasers/chars74k_teaser.png
   :align: center
   :width: 90%
   :alt: Chars74K EnglishHnd sample images


Data Structure
--------------

Each sample is returned as a dictionary:

.. list-table::
   :header-rows: 1

   * - Key
     - Type
     - Description
   * - ``image``
     - ``PIL.Image.Image``
     - RGB handwritten character image
   * - ``label``
     - int
     - Class index from 0 to 61


Label Mapping
-------------

The dataset uses 62 class folders, named
``Sample001`` through ``Sample062``.

- ``Sample001`` to ``Sample010``: digits 0-9
- ``Sample011`` to ``Sample036``: uppercase letters A-Z
- ``Sample037`` to ``Sample062``: lowercase letters a-z

The builder maps each class to the corresponding
zero-based class index.

Official labels are also checked against the image
folder names during dataset preparation.


Train/Test Split
----------------

This implementation uses the official Chars74K
EnglishHnd split definitions instead of a custom
train/test split.

The image archive is ``EnglishHnd.tgz``.
The official split definitions are stored in
``Lists.tgz``.

The builder reads the following MATLAB file:

``Lists/English/Hnd/lists_var_size.mat``

The file contains:

- ``ALLnames``: image filenames
- ``ALLlabels``: official class labels
- ``TRNind``: official training indices
- ``TSTind``: official testing indices

The official index arrays provide 30 split configurations.
This implementation selects the first split
(``OFFICIAL_SPLIT_INDEX = 0``).

The builder removes zero-padding entries and converts
MATLAB 1-based indices to Python 0-based indices.

For the selected split:

- Training samples: 310
- Test samples: 1,674
- Training classes: 62
- Test classes: 62
- Overlapping training/test samples: 0

The remaining images are not automatically reassigned
to either split.


Usage Example
-------------

.. code-block:: python

   from stable_datasets.images import Chars74K

   train_ds = Chars74K(split="train")
   test_ds = Chars74K(split="test")

   print(len(train_ds))  # 310
   print(len(test_ds))   # 1674

   sample = train_ds[0]
   image = sample["image"]
   label = sample["label"]

   print(image.size, image.mode)
   print(label)


Validation
----------

The dataset implementation was validated using
offline unit tests and real-data integration tests.

The tests cover:

- Label mapping across 62 classes
- Official MATLAB split parsing
- MATLAB zero-padding removal
- Image filename normalization
- Training and test sample generation
- Invalid image paths and missing split files
- Real-data loading through the public dataset API
- Official training and test sample counts
- Training/test index uniqueness and disjointness

The selected official split contains 310 training
samples and 1,674 test samples.

Both splits cover all 62 classes.
No overlapping image indices were found.

Run all tests, including real-data integration, with:

.. code-block:: bash

   pytest stable_datasets/tests/images/test_chars74k.py -v


The real-data tests require access to the official
Chars74K archives.


Citation
--------

If you use Chars74K in your research, please cite
the original dataset paper:

.. code-block:: bibtex

   @inproceedings{deCampos2009chars74k,
       title={Character Recognition in Natural Images},
       author={de Campos, Teofilo E. and Babu, Bodla Rakesh and Varma, Manik},
       booktitle={VISAPP},
       year={2009}
   }


References
----------

- `Chars74K dataset <https://teodecampos.github.io/chars74k/>`_
- `EnglishHnd archive <https://info-ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishHnd.tgz>`_
- `Official split archive <https://info-ee.surrey.ac.uk/CVSSP/demos/chars74k/Lists.tgz>`_
