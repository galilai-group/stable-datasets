Tiny ImageNet
==============

.. raw:: html

	 <p style="display: flex; gap: 10px;">
	 <img src="https://img.shields.io/badge/Task-Image%20Classification-blue" alt="Task: Image Classification">
	 <img src="https://img.shields.io/badge/Classes-200-green" alt="Classes: 200">
	 <img src="https://img.shields.io/badge/Size-64x64-orange" alt="Image Size: 64x64">
	 </p>

Overview
--------

Tiny ImageNet is a downsampled subset of the ImageNet ILSVRC dataset created for the Tiny ImageNet Visual Recognition Challenge. It contains 200 classes and approximately 100,000 images resized to 64×64 pixels in RGB format. For each class there are typically 500 training images and 50 validation images.

- **Train**: ~100,000 images total (500 images per class)
- **Validation**: 10,000 images (50 images per class)

Data Structure
--------------

When accessing an example using ``ds[i]`` you will receive a dictionary with these keys:

.. list-table::
	 :header-rows: 1
	 :widths: 20 20 60

	* - Key
	  - Type
	  - Description
	* - ``image``
	  - ``PIL.Image.Image``
	  - 64×64×3 RGB image
	* - ``label``
	  - int
	  - Class label (0-199) or string class id depending on dataset builder

Usage Example
-------------

**Basic Usage**

.. code-block:: text

		from stable_datasets.images.tiny_imagenet import TinyImagenet

		# First run will download + prepare cache, then return the split as a HF Dataset
		ds = TinyImagenet(split="train")

		# If you omit the split (split=None), you get a DatasetDict with all available splits
		ds_all = TinyImagenet(split=None)

		sample = ds[0]
		print(sample.keys())  # {"image", "label"}

		# Optional: make it PyTorch-friendly
		ds_torch = ds.with_format("torch")

Related Datasets
----------------

- `tiny_imagenet_c` : Corrupted version with common corruptions applied
- CIFAR variants: smaller benchmarks with similar tasks (see `cifar10`, `cifar100`)

References
----------

- Tiny ImageNet (CS231n page): http://cs231n.stanford.edu/tiny-imagenet-200.html
- Competition / resources: https://www.kaggle.com/c/tiny-imagenet

Citation
--------

.. code-block:: bibtex

		@inproceedings{Le2015TinyIV,
			title={Tiny ImageNet Visual Recognition Challenge},
			author={Ya Le and Xuan S. Yang},
			year={2015}
		}

