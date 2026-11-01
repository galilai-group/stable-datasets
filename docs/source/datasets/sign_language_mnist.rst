
Sign Language MNIST
===================

Overview
--------

Sign Language MNIST is an image classification dataset
containing grayscale images of American Sign Language
(ASL) hand gestures.

The dataset contains 24 classes corresponding to the
letters A-Z, excluding J and Z, whose gestures involve
motion.

Dataset statistics:

- Training samples: 27,455
- Test samples: 7,172
- Number of classes: 24
- Image resolution: 28 x 28
- Image format: grayscale
- Task: image classification

.. image:: teasers/sign_language_mnist_teaser.png
   :align: center
   :width: 90%
   :alt: Sign Language MNIST sample images


Data Source
-----------

The dataset is downloaded directly from the original
Sign Language MNIST dataset published on Kaggle.

The builder uses the official Kaggle Dataset Download API.

The downloaded ZIP archive contains the following CSV files:

- ``sign_mnist_train.csv``: training data
- ``sign_mnist_test.csv``: test data

The archive also includes copies of these CSV files
inside separate train and test directories.

The builder selects the root-level CSV files to
avoid processing duplicate data.

Each CSV row contains a label and 784 pixel values.
Pixel values are integers between 0 and 255.

The original training and test splits are preserved.
No custom split is introduced.


Data Structure
--------------

Each sample is returned as a dictionary with two fields:

.. list-table::
   :header-rows: 1

   * - Key
     - Type
     - Description
   * - ``image``
     - ``PIL.Image.Image``
     - 28 x 28 grayscale image
   * - ``label``
     - int
     - Class index in the range 0-23


Label Mapping
-------------

The original dataset uses letter indices 0-25,
but excludes indices 9 (J) and 25 (Z).

This builder maps the 24 existing classes to
contiguous indices 0-23 while preserving their
alphabetical order.

The original labels and their mapped values are
validated during dataset preparation.


Usage Example
-------------

.. code-block:: python

   from stable_datasets.images import SignLanguageMNIST

   train_ds = SignLanguageMNIST(split="train")
   test_ds = SignLanguageMNIST(split="test")

   print(len(train_ds))  # 27455
   print(len(test_ds))   # 7172

   sample = train_ds[0]
   image = sample["image"]
   label = sample["label"]

   print(image.size)  # (28, 28)
   print(image.mode)  # L
   print(label)


Validation
----------

The implementation was validated using offline unit tests
and real-data integration tests.

The offline tests cover:

- Label mapping across all 24 classes
- Train/test CSV selection
- CSV header validation
- Missing CSV files
- Invalid labels
- Invalid pixel values
- CSV-to-image conversion

The real-data integration test downloads the original
Kaggle dataset and validates:

- Training samples: 27,455
- Test samples: 7,172
- Number of classes: 24
- Image resolution: 28 x 28
- Image mode: grayscale
- Original CSV header and label set
- Loading through the public dataset API

Run all tests, including real-data integration, with:

.. code-block:: bash

   pytest stable_datasets/tests/images/test_sign_language_mnist.py -v


The real-data test requires network access to Kaggle.

The dataset successfully passed all 14 tests,
including the real-data integration test.


License
-------

The original Sign Language MNIST dataset is published
on Kaggle under the CC0: Public Domain license.

For the original license information, refer to
the Kaggle dataset page.


Citation
--------

If you use Sign Language MNIST in your research,
please reference the original Kaggle dataset.

.. code-block:: bibtex

   @misc{sign_language_mnist,
       title={Sign Language MNIST},
       author={{DataMunge}},
       howpublished={Kaggle dataset},
       url={https://www.kaggle.com/datasets/datamunge/sign-language-mnist}
   }


References
----------

- `Sign Language MNIST (Kaggle) <https://www.kaggle.com/datasets/datamunge/sign-language-mnist>`_
- `Kaggle Dataset Download API <https://www.kaggle.com/api/v1/datasets/download/datamunge/sign-language-mnist>`_
