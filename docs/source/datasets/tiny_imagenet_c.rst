Tiny ImageNet-C
===============

.. raw:: html

	 <p style="display: flex; gap: 10px;">
	 <img src="https://img.shields.io/badge/Task-Image%20Classification-blue" alt="Task: Image Classification">
	 <img src="https://img.shields.io/badge/Classes-200-green" alt="Classes: 200">
	 <img src="https://img.shields.io/badge/Corruptions-Multiple-orange" alt="Corruptions: Multiple">
	 </p>

Overview
--------

Tiny ImageNet-C is a corruption-benchmarked variant of Tiny ImageNet created by applying the common corruptions and perturbations introduced by Hendrycks & Dietterich (2019). The corrupted dataset is intended to evaluate model robustness to naturalistic corruptions: each image is processed with a corruption type at multiple severity levels.

- **Classes**: 200 (same label set as Tiny ImageNet)
- **Corruptions**: multiple corruption types (see the original ImageNet-C paper for the canonical list)
- **Severity levels**: multiple levels (commonly 5 severity levels)

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
     - Corrupted RGB image (64×64)
   * - ``label``
     - int
     - Class label (0-199) or string class id depending on dataset builder
   * - ``corruption_name``
     - ``str``
     - Name of the corruption applied (e.g., gaussian_noise, motion_blur)
   * - ``corruption_level``
     - ``int``
     - Severity level for the corruption (typically 1-5)

Usage Example
-------------

.. code-block:: text

	from stable_datasets.images.tiny_imagenet_c import TinyImagenetC

	# Loads the corrupted test split (download + prepare happens on first call)
	ds = TinyImagenetC(split="test")

	sample = ds[0]
	print(sample.keys())  # {"image", "label", "corruption_name", "corruption_level"}

Related Datasets
----------------

- `tiny_imagenet` : The original (clean) Tiny ImageNet dataset
- ImageNet-C: larger corruption benchmark for ImageNet (Hendrycks et al.)

References
----------

- Hendrycks, D. and Dietterich, T. (2019). Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. ICLR.
- Zenodo archive: https://zenodo.org/records/2536630

Citation
--------

.. code-block:: text

		@article{hendrycks2019robustness,
			title={Benchmarking Neural Network Robustness to Common Corruptions and Perturbations},
			author={Hendrycks, Dan and Dietterich, Thomas},
			journal={Proceedings of the International Conference on Learning Representations},
			year={2019}
		}

