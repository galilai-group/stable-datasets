not-MNIST
=========

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Image%20Classification-blue" alt="Task: Image Classification">
   <img src="https://img.shields.io/badge/Classes-10-green" alt="Classes: 10">
   <img src="https://img.shields.io/badge/Size-28x28-orange" alt="Image Size: 28x28">
   <img src="https://img.shields.io/badge/Format-Grayscale-lightgrey" alt="Format: Grayscale">
   </p>

Overview
--------

The **not-MNIST** dataset was created by Yaroslav Bulatov as a more challenging alternative to the classic MNIST dataset. While the original MNIST consists of handwritten digits, not-MNIST is composed of glyphs extracted from various publicly available fonts.

This dataset features the letters **A through J** (10 classes) and serves as a rigorous benchmark for machine learning models. It is significantly more difficult than MNIST because the fonts range from standard typefaces to highly artistic, experimental, or even barely legible designs.

Split sizes:

- **Train**: 60,000 images
- **Test**: 10,000 images

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
     - 28×28 Grayscale image of a letter (A-J)
   * - ``label``
     - int
     - Class label (0-9) corresponding to letters A through J

Usage Example
-------------

**Basic Usage**

.. code-block:: python

    from stable_datasets.images.not_mnist import NotMNIST

    # First run will download + prepare cache, then return the split as a HF Dataset
    ds = NotMNIST(split="train")

    # If you omit the split (split=None), you get a DatasetDict with all available splits
    ds_all = NotMNIST(split=None)

    sample = ds[0]
    print(sample.keys())  # {"image", "label"}

    # Access the image and label
    image = sample["image"]  # PIL.Image.Image
    label = sample["label"]  # int (0-9)
    print(f"Label: {label} -> Letter: {chr(ord('A') + label)}")

    # Optional: make it PyTorch-friendly
    ds_torch = ds.with_format("torch")

**With Transforms**

.. code-block:: python

    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    ds = NotMNIST(split="train")
    sample = ds[0]
    tensor = transform(sample["image"])
    print(f"Tensor shape: {tensor.shape}")  # torch.Size([1, 28, 28])

Related Datasets
----------------

- :doc:`mnist`: The original handwritten digit dataset
- **EMNIST**: Extended MNIST including both digits and handwritten letters
- **Fashion-MNIST**: A replacement for MNIST consisting of clothing items

References
----------

- Creator: Yaroslav Bulatov
- Blog: http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html
- Data source: https://github.com/davidflanagan/notMNIST-to-MNIST

Citation
--------

.. code-block:: bibtex

    @misc{bulatov2011notmnist,
        author={Yaroslav Bulatov},
        title={notMNIST dataset},
        year={2011},
        url={http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html}
    }
